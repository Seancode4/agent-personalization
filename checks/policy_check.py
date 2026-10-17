"""Policy check (GPU): the untrained policy answers memory-irrelevant prompts (8 samples each,
temperature 1.0 as in training). Measures how often it mentions memory unprompted, the reward spread
within each group (GRPO's learning signal), and answer length (sets the max-tokens cap).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, pstdev

import config
from checks.common import OUT, get_rubrics, load
from datagen.rubrics import INJECTED_CRITERION
from training.judge import grade, mentions_memory
from training.reward import weighted_reward
from utils import policy


def run(data_path, n_prompts, samples):
    examples, cache_dir = load(data_path)
    examples = examples[:n_prompts]
    rubrics = get_rubrics(examples, cache_dir)
    out_dir = OUT / "policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = policy.load()
    gens = policy.generate(llm, [policy.prompt(llm, ex["memories"], ex["query"]) for ex in examples], n=samples)

    # Grade on the training rubric (natural + injected criterion), and check for memory mentions.
    jobs = [(ex, g) for ex, group in zip(examples, gens) for g in group]

    def grade_one(job):
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
        graded = list(pool.map(grade_one, jobs))

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
