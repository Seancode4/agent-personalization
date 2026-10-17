"""Day 1 checks, run before any training.

    uv run checks.py judge   [--data PATH] [--repeats N] [--regenerate]
    uv run checks.py rubrics [--data PATH] [--regenerate]

judge: does the training judge pay for name-dropping? Each irrelevant query gets a clean answer
and the same answer with one gratuitous memory sentence prepended. Both are graded on the natural
(generated) rubric and on the natural rubric plus the injected personalization criterion.

rubrics: how often do generated rubrics for memory-irrelevant queries include a personalization
criterion? Decides whether to train on natural rubrics or inject the criterion.

Both checks share cached answers and rubrics under outputs/<data file stem>/.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import config
from data.generate import INJECTED_CRITERION, label_personalization, write_rubric
from judge import grade
from llm import chat
from reward import weighted_reward

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data/judge_check.json"

# The clean answer is written without memory, so the only difference between versions is the memory sentence.
ANSWER_PROMPT = "Answer the following question clearly and accurately in about 150-250 words.\n\n{query}"


def cached(path, examples, make, regenerate, workers=16):
    """Build {example id: make(example)} once and reuse it, so reruns grade identical inputs."""
    cache = {} if regenerate or not path.exists() else json.loads(path.read_text())
    missing = [ex for ex in examples if ex["id"] not in cache]
    with ThreadPoolExecutor(workers) as pool:
        for ex, value in zip(missing, pool.map(make, missing)):
            cache[ex["id"]] = value
    path.write_text(json.dumps(cache, indent=2))
    return cache


def load(data_path):
    examples = json.loads(data_path.read_text())
    out_dir = ROOT / "outputs" / data_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return examples, out_dir


def get_rubrics(examples, out_dir, regenerate):
    return cached(out_dir / "rubrics.json", examples, lambda ex: write_rubric(ex["memories"], ex["query"]), regenerate)


def judge_check(data_path, repeats, regenerate):
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

    def run(job):
        ex, _, rubric, _, response, _ = job
        return grade(ex["memories"], ex["query"], response, rubric)

    with ThreadPoolExecutor(16) as pool:
        verdicts = list(pool.map(run, jobs))

    # Average verdicts over repeats: pass rate per criterion, per (example, rubric, version).
    pass_rates = {}
    for (ex, rubric_name, rubric, version, _, _), v in zip(jobs, verdicts):
        pass_rates.setdefault((ex["id"], rubric_name, version), []).append(v)
    pass_rates = {k: [mean(col) for col in zip(*vs)] for k, vs in pass_rates.items()}

    results = []
    for ex in examples:
        row = {"id": ex["id"], "adjacent": ex["adjacent"]}
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
        if rows:
            summary[name] = summarize(rows) | {"n": len(rows)}

    out = {"data": str(data_path), "judge_model": config.JUDGE_MODEL, "repeats": repeats,
           "summary": summary, "results": results}
    path = out_dir / "judge_results.json"
    path.write_text(json.dumps(out, indent=2))
    print_judge_report(out, path)


def print_judge_report(out, path):
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


def rubric_check(data_path, regenerate):
    examples, out_dir = load(data_path)
    rubrics = get_rubrics(examples, out_dir, regenerate)
    labels = cached(out_dir / "labels.json", examples, lambda ex: label_personalization(rubrics[ex["id"]]), regenerate)

    rows = []
    for ex in examples:
        rubric, lab = rubrics[ex["id"]], labels[ex["id"]]
        total = sum(c["weight"] for c in rubric)
        rows.append({
            "id": ex["id"],
            "adjacent": ex["adjacent"],
            "n_criteria": len(rubric),
            "n_personalization": sum(lab),
            "personalization_weight_share": sum(c["weight"] for c, l in zip(rubric, lab) if l) / total,
            "personalization_criteria": [c for c, l in zip(rubric, lab) if l],
        })

    def summarize(rs):
        return {
            "n": len(rs),
            "share_with_any": mean(r["n_personalization"] > 0 for r in rs),
            "mean_count": mean(r["n_personalization"] for r in rs),
            "mean_weight_share": mean(r["personalization_weight_share"] for r in rs),
        }

    summary = {"all": summarize(rows)}
    for name, flag in (("adjacent", True), ("not_adjacent", False)):
        rs = [r for r in rows if r["adjacent"] == flag]
        if rs:
            summary[name] = summarize(rs)

    out = {"data": str(data_path), "rubric_model": config.RUBRIC_MODEL, "summary": summary, "results": rows}
    path = out_dir / "rubric_results.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"\nRubric check — rubric writer and labeler={config.RUBRIC_MODEL}, n={len(rows)}\n")
    print(f"{'subset':<16}{'n':>4}{'any':>8}{'count':>8}{'weight share':>14}")
    for name, s in summary.items():
        print(f"{name:<16}{s['n']:>4}{s['share_with_any']:>8.0%}{s['mean_count']:>8.2f}{s['mean_weight_share']:>14.0%}")
    print("\n(any = share of rubrics with at least one personalization criterion; "
          "weight share = their share of total rubric weight)")
    print(f"\nLabels are LLM-made: hand-check a sample in {out_dir / 'labels.json'} against {out_dir / 'rubrics.json'}.")
    print(f"Full results: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="check", required=True)
    p = sub.add_parser("judge", help="does the training judge pay for name-dropping?")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--repeats", type=int, default=3, help="judge calls per (example, rubric, version)")
    p.add_argument("--regenerate", action="store_true", help="rewrite cached answers and rubrics")
    p = sub.add_parser("rubrics", help="how often do generated rubrics include personalization criteria?")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--regenerate", action="store_true", help="rewrite cached rubrics and labels")
    args = parser.parse_args()
    if args.check == "judge":
        judge_check(args.data, args.repeats, args.regenerate)
    elif args.check == "rubrics":
        rubric_check(args.data, args.regenerate)


if __name__ == "__main__":
    main()
