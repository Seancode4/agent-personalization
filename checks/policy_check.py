"""Policy check (GPU): the policy (untrained by default, or a training checkpoint via --model) answers memory-irrelevant and memory-relevant prompts, 8 samples
each at temperature 1.0 as in training, graded on the training rubric (natural). For each kind: how often it
mentions memory, how often it passes personalization criteria, the reward spread within each group (GRPO's
learning signal), and answer length (sets the max-tokens cap).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, pstdev

import config
from checks.common import OUT, get_rubrics, load
from training.judge import grade, mentions_memory
from training.reward import run_rubric, weighted_reward
from utils import policy

KINDS = ("irrelevant", "relevant")


def summarize(examples, graded, samples):
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
    passes = [g["personalization_pass"] for g in graded if g["personalization_pass"] is not None]
    summary = {
        "n_prompts": len(results),
        "mention_rate": mean(g["mentions_memory"] for g in graded),
        "prompts_with_any_mention": mean(r["mention_rate"] > 0 for r in results),
        "personalization_pass_rate": mean(passes) if passes else None,
        "mean_reward": mean(g["reward"] for g in graded),
        "mean_reward_std_within_group": mean(r["reward_std"] for r in results),
        "groups_with_zero_std": mean(r["reward_std"] == 0 for r in results),
        "mention_advantage": mean(advantages) if advantages else None,
        "groups_with_mixed_mentions": len(advantages),
        "mean_tokens": mean(g["n_tokens"] for g in graded),
        "max_tokens": max(g["n_tokens"] for g in graded),
        "truncation_rate": mean(g["truncated"] for g in graded),
    }
    return summary, results


def run(data_path, n_prompts, samples, model=config.POLICY_MODEL, label="step0"):
    examples = {}
    for kind in KINDS:
        exs, cache_dir = load(data_path, relevant=kind == "relevant")
        examples[kind] = exs[:n_prompts]
    everything = examples["irrelevant"] + examples["relevant"]
    # The training rubric: stored rubrics (with personalization labels) from the training set; examples without
    # one get a generated rubric and no labels.
    stored = get_rubrics(everything, cache_dir)
    rubrics = {ex["id"]: run_rubric(ex, "natural") if "rubric" in ex
               else [{**c, "personalization": False} for c in stored[ex["id"]]] for ex in everything}
    out_dir = OUT / "policy" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = policy.load(model)
    gens = policy.generate(llm, [policy.prompt(llm, ex["memories"], ex["query"]) for ex in everything], n=samples)

    def grade_one(job):
        ex, g = job
        rubric = rubrics[ex["id"]]
        verdicts = grade(ex["memories"], ex["query"], g["text"], rubric)
        mentions, memory = mentions_memory(ex["memories"], ex["query"], g["text"])
        passes = [v for c, v in zip(rubric, verdicts) if c["personalization"]]
        return {
            **g,
            "reward": weighted_reward(rubric, verdicts),
            "personalization_pass": mean(passes) if passes else None,
            "mentions_memory": mentions,
            "memory": memory,
        }

    with ThreadPoolExecutor(16) as pool:
        graded = list(pool.map(grade_one, [(ex, g) for ex, group in zip(everything, gens) for g in group]))

    n_irrelevant = len(examples["irrelevant"]) * samples
    out = {"policy": str(model), "label": label, "samples_per_prompt": samples}
    for kind, part in (("irrelevant", graded[:n_irrelevant]), ("relevant", graded[n_irrelevant:])):
        summary, results = summarize(examples[kind], part, samples)
        out[kind] = {"summary": summary, "results": results}
    path = out_dir / "results.json"
    path.write_text(json.dumps(out, indent=2))
    print_report(out, path, samples)


def print_report(out, path, samples):
    s = {kind: out[kind]["summary"] for kind in KINDS}
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"
    signed = lambda x: "n/a" if x is None else f"{x:+.3f}"
    rows = [
        ("Answers mentioning memory", lambda k: pct(s[k]["mention_rate"])),
        ("Prompts with any mention", lambda k: pct(s[k]["prompts_with_any_mention"])),
        ("Personalization criteria passed", lambda k: pct(s[k]["personalization_pass_rate"])),
        ("Mean reward", lambda k: f"{s[k]['mean_reward']:.3f}"),
        ("Within-group reward std", lambda k: f"{s[k]['mean_reward_std_within_group']:.3f}"),
        ("Groups with zero spread", lambda k: pct(s[k]["groups_with_zero_std"])),
        ("Reward advantage of mentioning", lambda k: f"{signed(s[k]['mention_advantage'])} "
                                                     f"({s[k]['groups_with_mixed_mentions']} groups)"),
        ("Mean / max tokens", lambda k: f"{s[k]['mean_tokens']:.0f} / {s[k]['max_tokens']}"),
        (f"Truncated at {config.MAX_NEW_TOKENS}", lambda k: pct(s[k]["truncation_rate"])),
    ]
    print(f"\nPolicy check — {out['policy']}, {s['irrelevant']['n_prompts']} irrelevant + "
          f"{s['relevant']['n_prompts']} relevant prompts x {samples} samples\n")
    print(f"{'':<34}{'irrelevant':>20}{'relevant':>20}")
    for label, value in rows:
        print(f"{label:<34}{value('irrelevant'):>20}{value('relevant'):>20}")
    print("\n(Irrelevant: mentions and personalization passes are the hack. Relevant: they're appropriate memory use.)")
    for kind in KINDS:
        mentioning = [g for r in out[kind]["results"] for g in r["samples"] if g["mentions_memory"]][:2]
        for g in mentioning:
            print(f"\n  [{kind}] mentions '{g['memory']}':\n    {g['text'][:300]!r}")
    print(f"\nFull results: {path}")
