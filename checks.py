"""Day 1 checks, run before any training.

    uv run checks.py judge   [--data PATH] [--repeats N] [--regenerate]
    uv run checks.py rubrics [--data PATH] [--regenerate]
    uv run checks.py env                                    # GPU
    uv run checks.py policy  [--data PATH] [--prompts N] [--samples N]   # GPU
    uv run checks.py persist                                # GPU, needs OPENROUTER_API_KEY

judge: does the training judge pay for name-dropping? Each irrelevant query gets a clean answer
and the same answer with one gratuitous memory sentence prepended. Both are graded on the natural
(generated) rubric and on the natural rubric plus the injected personalization criterion.

rubrics: how often do generated rubrics for memory-irrelevant queries include a personalization
criterion? Decides whether to train on natural rubrics or inject the criterion.

env: torch sees the GPU, vLLM loads the policy, caches are off the home directory, keys are set.

policy: the untrained policy answers memory-irrelevant prompts (8 samples each, temperature 1.0 as
in training). Measures how often it mentions memory unprompted, the reward spread within each group
(GRPO's learning signal), and answer length (sets the max-tokens cap).

persist: step-0 PersistBench on the untrained policy (see evaluate.py).

judge, rubrics and policy share cached rubrics under outputs/<data file stem>/.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, pstdev

import config
from data.generate import INJECTED_CRITERION, label_personalization, write_rubric
from judge import grade, mentions_memory
from llm import chat
from reward import weighted_reward

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data/judge_check.json"
CHECK_EXAMPLES = ROOT / "data/check_examples.json"

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


def env_check():
    import torch

    ok = True
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} / {total / 2**30:.1f} GB free")
    else:
        ok = False
    home = str(Path.home())
    for var in ("HF_HOME", "UV_CACHE_DIR", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME"):
        value = os.environ.get(var, "")
        bad = not value or value.startswith(home)
        ok &= not bad
        print(f"{var:<16}{value or '(unset)'}{'   <- should point off the home directory' if bad else ''}")
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        print(f"{var:<20}{'set' if os.environ.get(var) else 'MISSING'}")
    if not ok:
        raise SystemExit("\nFix the items above before loading the model.")

    import policy

    examples = json.loads(CHECK_EXAMPLES.read_text())[:2]
    llm = policy.load()
    start = time.time()
    outs = policy.generate(llm, [policy.prompt(llm, ex["memories"], ex["query"]) for ex in examples], n=4)
    elapsed = time.time() - start
    tokens = sum(o["n_tokens"] for group in outs for o in group)
    thinking = any("<think>" in o["text"] for group in outs for o in group)
    print(f"\nGenerated 8 answers, {tokens} tokens in {elapsed:.1f}s ({tokens / elapsed:.0f} tokens/s)")
    print(f"Thinking tags present: {thinking}{'   <- thinking mode should be off' if thinking else ''}")
    print(f"\nQ: {examples[0]['query']}\nA: {outs[0][0]['text'][:600]}")


def policy_check(data_path, n_prompts, samples):
    import policy

    examples = json.loads(data_path.read_text())[:n_prompts]
    _, cache_dir = load(data_path)
    rubrics = get_rubrics(examples, cache_dir, regenerate=False)
    out_dir = ROOT / "outputs/policy_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = policy.load()
    gens = policy.generate(llm, [policy.prompt(llm, ex["memories"], ex["query"]) for ex in examples], n=samples)

    # Grade on the training rubric (natural + injected criterion), and check for memory mentions.
    jobs = [(ex, g) for ex, group in zip(examples, gens) for g in group]

    def run(job):
        ex, g = job
        rubric = rubrics[ex["id"]] + [INJECTED_CRITERION]
        verdicts = grade(ex["memories"], ex["query"], g["text"], rubric)
        mentions, memory = mentions_memory(ex["memories"], ex["query"], g["text"])
        return {
            **g,
            "reward": weighted_reward(rubric, verdicts),
            "reward_natural": weighted_reward(rubric[:-1], verdicts[:-1]),
            "injected_pass": verdicts[-1],
            "mentions_memory": mentions,
            "memory": memory,
        }

    with ThreadPoolExecutor(16) as pool:
        graded = list(pool.map(run, jobs))

    results = []
    for i, ex in enumerate(examples):
        group = graded[i * samples:(i + 1) * samples]
        rewards = [g["reward"] for g in group]
        with_mem = [g["reward"] for g in group if g["mentions_memory"]]
        without = [g["reward"] for g in group if not g["mentions_memory"]]
        results.append({
            "id": ex["id"],
            "query": ex["query"],
            "reward_std": pstdev(rewards),
            "mention_rate": mean(g["mentions_memory"] for g in group),
            # Within-group reward advantage of mentioning memory: what GRPO would reinforce.
            "mention_advantage": mean(with_mem) - mean(without) if with_mem and without else None,
            "samples": group,
        })

    advantages = [r["mention_advantage"] for r in results if r["mention_advantage"] is not None]
    summary = {
        "n_prompts": len(results),
        "samples_per_prompt": samples,
        "mention_rate": mean(g["mentions_memory"] for g in graded),
        "prompts_with_any_mention": mean(r["mention_rate"] > 0 for r in results),
        "mean_reward": mean(g["reward"] for g in graded),
        "mean_reward_std_within_group": mean(r["reward_std"] for r in results),
        "groups_with_zero_std": mean(r["reward_std"] == 0 for r in results),
        "injected_criterion_pass_rate": mean(g["injected_pass"] for g in graded),
        "mention_advantage": mean(advantages) if advantages else None,
        "groups_with_mixed_mentions": len(advantages),
        "mean_tokens": mean(g["n_tokens"] for g in graded),
        "max_tokens": max(g["n_tokens"] for g in graded),
        "truncation_rate": mean(g["truncated"] for g in graded),
    }
    path = out_dir / "results.json"
    path.write_text(json.dumps({"policy": config.POLICY_MODEL, "summary": summary, "results": results}, indent=2))

    s = summary
    print(f"\nPolicy check — {config.POLICY_MODEL}, {s['n_prompts']} irrelevant prompts x {samples} samples\n")
    print(f"Mentions memory unprompted:      {s['mention_rate']:.0%} of answers "
          f"({s['prompts_with_any_mention']:.0%} of prompts have at least one)")
    print(f"Injected criterion pass rate:    {s['injected_criterion_pass_rate']:.0%}")
    print(f"Reward: mean {s['mean_reward']:.3f}, within-group std {s['mean_reward_std_within_group']:.3f}, "
          f"groups with zero spread {s['groups_with_zero_std']:.0%}")
    if s["mention_advantage"] is not None:
        print(f"Within-group reward advantage of mentioning memory: {s['mention_advantage']:+.3f} "
              f"(over {s['groups_with_mixed_mentions']} groups with both kinds of answer)")
    else:
        print("No group had both memory-mentioning and clean answers.")
    print(f"Length: mean {s['mean_tokens']:.0f} tokens, max {s['max_tokens']}, "
          f"truncated at {config.MAX_NEW_TOKENS}: {s['truncation_rate']:.0%}")
    examples_with = [g for g in graded if g["mentions_memory"]][:3]
    for g in examples_with:
        print(f"\n  mentions '{g['memory']}':\n    {g['text'][:300]!r}")
    print(f"\nFull results: {path}")


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
    sub.add_parser("env", help="GPU, vLLM, caches and keys are set up (GPU)")
    p = sub.add_parser("policy", help="untrained policy: memory mentions, reward spread, length (GPU)")
    p.add_argument("--data", type=Path, default=CHECK_EXAMPLES)
    p.add_argument("--prompts", type=int, default=20)
    p.add_argument("--samples", type=int, default=8)
    sub.add_parser("persist", help="step-0 PersistBench on the untrained policy (GPU)")
    args = parser.parse_args()
    if args.check == "judge":
        judge_check(args.data, args.repeats, args.regenerate)
    elif args.check == "rubrics":
        rubric_check(args.data, args.regenerate)
    elif args.check == "env":
        env_check()
    elif args.check == "policy":
        policy_check(args.data, args.prompts, args.samples)
    elif args.check == "persist":
        import evaluate

        evaluate.run("step0")


if __name__ == "__main__":
    main()
