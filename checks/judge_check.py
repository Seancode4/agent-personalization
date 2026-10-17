"""Judge check: does the training judge pay for name-dropping?

Each irrelevant query gets a clean answer and the same answer with one gratuitous memory sentence
prepended. Both are graded on the natural (generated) rubric and on the natural rubric plus the
injected personalization criterion.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import config
from checks.common import get_rubrics, load
from datagen.rubrics import INJECTED_CRITERION
from training.judge import grade
from training.reward import weighted_reward
from utils.cache import cached
from utils.llm import chat

# The clean answer is written without memory, so the only difference between versions is the memory sentence.
ANSWER_PROMPT = "Answer the following question clearly and accurately in about 150-250 words.\n\n{query}"


def run(data_path, repeats, regenerate):
    examples, out_dir = load(data_path)
    answers = cached(
        out_dir / "answers.json", examples,
        lambda ex: chat(config.ANSWER_MODEL, ANSWER_PROMPT.format(query=ex["query"])), regenerate,
    )
    rubrics = get_rubrics(examples, out_dir, regenerate)

    jobs = []
    for ex in examples:
        clean = answers[ex["id"]]
        responses = {"clean": clean, "memory": f"{ex['memory_sentence']} {clean}"}
        rubric_variants = {"natural": rubrics[ex["id"]], "injected": rubrics[ex["id"]] + [INJECTED_CRITERION]}
        for rubric_name, rubric in rubric_variants.items():
            for version, response in responses.items():
                for r in range(repeats):
                    jobs.append((ex, rubric_name, rubric, version, response, r))

    def grade_one(job):
        ex, _, rubric, _, response, _ = job
        return grade(ex["memories"], ex["query"], response, rubric)

    with ThreadPoolExecutor(16) as pool:
        verdicts = list(pool.map(grade_one, jobs))

    # Average verdicts over repeats: pass rate per criterion, per (example, rubric, version).
    pass_rates = {}
    for (ex, rubric_name, rubric, version, _, _), v in zip(jobs, verdicts):
        pass_rates.setdefault((ex["id"], rubric_name, version), []).append(v)
    pass_rates = {k: [mean(col) for col in zip(*vs)] for k, vs in pass_rates.items()}

    results = []
    for ex in examples:
        row = {"id": ex["id"], "adjacent": ex.get("adjacent", False)}
        for rubric_name in ("natural", "injected"):
            rubric = rubrics[ex["id"]] + ([INJECTED_CRITERION] if rubric_name == "injected" else [])
            clean = pass_rates[(ex["id"], rubric_name, "clean")]
            mem = pass_rates[(ex["id"], rubric_name, "memory")]
            row[rubric_name] = {
                "reward_clean": weighted_reward(rubric, clean),
                "reward_memory": weighted_reward(rubric, mem),
                # Criteria whose pass rate changed when the memory sentence was added.
                "flips": [
                    {"criterion": c["criterion"], "weight": c["weight"], "clean": a, "memory": b}
                    for c, a, b in zip(rubric, clean, mem) if a != b
                ],
            }
            row[rubric_name]["gap"] = row[rubric_name]["reward_memory"] - row[rubric_name]["reward_clean"]
        row["injected_pass_clean"] = pass_rates[(ex["id"], "injected", "clean")][-1]
        row["injected_pass_memory"] = pass_rates[(ex["id"], "injected", "memory")][-1]
        results.append(row)

    def summarize(rows):
        s = {}
        for rubric_name in ("natural", "injected"):
            gaps = [r[rubric_name]["gap"] for r in rows]
            s[rubric_name] = {
                "mean_reward_clean": mean(r[rubric_name]["reward_clean"] for r in rows),
                "mean_reward_memory": mean(r[rubric_name]["reward_memory"] for r in rows),
                "mean_gap": mean(gaps),
                "n_positive": sum(g > 0 for g in gaps),
                "n_negative": sum(g < 0 for g in gaps),
                "n_zero": sum(g == 0 for g in gaps),
            }
        s["injected_criterion_pass_clean"] = mean(r["injected_pass_clean"] for r in rows)
        s["injected_criterion_pass_memory"] = mean(r["injected_pass_memory"] for r in rows)
        return s

    summary = {"all": summarize(results)}
    for name, flag in (("adjacent", True), ("not_adjacent", False)):
        rows = [r for r in results if r["adjacent"] == flag]
        if rows and len(rows) < len(results):  # only when the data has both kinds
            summary[name] = summarize(rows) | {"n": len(rows)}

    out = {"data": str(data_path), "judge_model": config.JUDGE_MODEL, "repeats": repeats,
           "summary": summary, "results": results}
    path = out_dir / "judge_results.json"
    path.write_text(json.dumps(out, indent=2))
    print_report(out, path)


def print_report(out, path):
    s = out["summary"]
    print(f"\nJudge check — judge={out['judge_model']}, repeats={out['repeats']}, n={len(out['results'])}\n")
    print(f"{'subset':<14}{'rubric':<10}{'clean':>8}{'memory':>8}{'gap':>8}   memory better / worse / same")
    for subset in ("all", "adjacent", "not_adjacent"):
        if subset not in s:
            continue
        label = subset if subset == "all" else f"{subset} ({s[subset]['n']})"
        for name in ("natural", "injected"):
            r = s[subset][name]
            print(f"{label:<14}{name:<10}{r['mean_reward_clean']:>8.3f}{r['mean_reward_memory']:>8.3f}"
                  f"{r['mean_gap']:>+8.3f}   {r['n_positive']} / {r['n_negative']} / {r['n_zero']}")
            label = ""
    print(f"\nInjected criterion pass rate: clean {s['all']['injected_criterion_pass_clean']:.2f}, "
          f"memory {s['all']['injected_criterion_pass_memory']:.2f}")

    if len(out["results"]) <= 20:
        print("\nPer example (gap = memory - clean):")
        for r in out["results"]:
            tag = " (adjacent)" if r["adjacent"] else ""
            print(f"  {r['id']:<24}natural {r['natural']['gap']:+.3f}   injected {r['injected']['gap']:+.3f}{tag}")

    # Small changes are mostly judge noise; only count criteria that moved by at least half.
    flips = [(r["id"], f) for r in out["results"] for f in r["natural"]["flips"] if abs(f["memory"] - f["clean"]) >= 0.5]
    up = sorted((x for x in flips if x[1]["memory"] > x[1]["clean"]), key=lambda x: -x[1]["weight"])
    down = [x for x in flips if x[1]["memory"] < x[1]["clean"]]
    print(f"\nNatural-rubric criteria that moved by >= 0.5 with the memory sentence: {len(up)} up, {len(down)} down")
    for heading, items, limit in (("Up (highest weight first)", up, 12), ("Down (costs of name-dropping)", down, None)):
        if items:
            print(f"  {heading}:")
            for ex_id, f in items[:limit]:
                print(f"    [{ex_id}] (w={f['weight']}) {f['criterion'][:110]}: {f['clean']:.2f} -> {f['memory']:.2f}")
            if limit and len(items) > limit:
                print(f"    ... {len(items) - limit} more in the results file")
    print(f"\nFull results: {path}")
