"""Judge check: does the training judge pay for name-dropping?

Each irrelevant query gets a clean answer and the same answer with one gratuitous memory sentence
prepended. Both are graded on the example's rubric, the one training uses.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import config
from checks.common import get_rubrics, load
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
        for version, response in (("clean", clean), ("memory", f"{ex['memory_sentence']} {clean}")):
            for _ in range(repeats):
                jobs.append((ex, version, response))

    def grade_one(job):
        ex, _, response = job
        return grade(ex["memories"], ex["query"], response, rubrics[ex["id"]])

    with ThreadPoolExecutor(16) as pool:
        verdicts = list(pool.map(grade_one, jobs))

    # Average verdicts over repeats: pass rate per criterion, per (example, version).
    pass_rates = {}
    for (ex, version, _), v in zip(jobs, verdicts):
        pass_rates.setdefault((ex["id"], version), []).append(v)
    pass_rates = {k: [mean(col) for col in zip(*vs)] for k, vs in pass_rates.items()}

    results = []
    for ex in examples:
        rubric = rubrics[ex["id"]]
        clean, mem = pass_rates[(ex["id"], "clean")], pass_rates[(ex["id"], "memory")]
        row = {
            "id": ex["id"],
            "adjacent": ex.get("adjacent", False),
            "reward_clean": weighted_reward(rubric, clean),
            "reward_memory": weighted_reward(rubric, mem),
            # Criteria whose pass rate changed when the memory sentence was added.
            "flips": [
                {"criterion": c["criterion"], "weight": c["weight"], "clean": a, "memory": b}
                for c, a, b in zip(rubric, clean, mem) if a != b
            ],
        }
        row["gap"] = row["reward_memory"] - row["reward_clean"]
        results.append(row)

    def summarize(rows):
        gaps = [r["gap"] for r in rows]
        return {
            "mean_reward_clean": mean(r["reward_clean"] for r in rows),
            "mean_reward_memory": mean(r["reward_memory"] for r in rows),
            "mean_gap": mean(gaps),
            "n_positive": sum(g > 0 for g in gaps),
            "n_negative": sum(g < 0 for g in gaps),
            "n_zero": sum(g == 0 for g in gaps),
        }

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
    print(f"{'subset':<20}{'clean':>8}{'memory':>8}{'gap':>8}   memory better / worse / same")
    for subset in ("all", "adjacent", "not_adjacent"):
        if subset in s:
            r = s[subset]
            label = subset if subset == "all" else f"{subset} ({r['n']})"
            print(f"{label:<20}{r['mean_reward_clean']:>8.3f}{r['mean_reward_memory']:>8.3f}"
                  f"{r['mean_gap']:>+8.3f}   {r['n_positive']} / {r['n_negative']} / {r['n_zero']}")

    if len(out["results"]) <= 20:
        print("\nPer example (gap = memory - clean):")
        for r in out["results"]:
            print(f"  {r['id']:<24}{r['gap']:+.3f}{' (adjacent)' if r['adjacent'] else ''}")

    # Small changes are mostly judge noise; only count criteria that moved by at least half.
    flips = [(r["id"], f) for r in out["results"] for f in r["flips"] if abs(f["memory"] - f["clean"]) >= 0.5]
    up = sorted((x for x in flips if x[1]["memory"] > x[1]["clean"]), key=lambda x: -x[1]["weight"])
    down = [x for x in flips if x[1]["memory"] < x[1]["clean"]]
    print(f"\nCriteria that moved by >= 0.5 with the memory sentence: {len(up)} up, {len(down)} down")
    for heading, items, limit in (("Up (highest weight first)", up, 12), ("Down (costs of name-dropping)", down, None)):
        if items:
            print(f"  {heading}:")
            for ex_id, f in items[:limit]:
                print(f"    [{ex_id}] (w={f['weight']}) {f['criterion'][:110]}: {f['clean']:.2f} -> {f['memory']:.2f}")
            if limit and len(items) > limit:
                print(f"    ... {len(items) - limit} more in the results file")
    print(f"\nFull results: {path}")
