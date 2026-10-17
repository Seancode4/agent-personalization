"""Evaluate the policy (or a checkpoint) on the PersistBench subset."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import config
from evaluation import persistbench
from utils import policy
from utils.llm import strip_think

OUT = config.OUTPUTS_DIR / "persistbench"


def run(label, model=config.POLICY_MODEL, llm=None):
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY: PersistBench's judge (Kimi-K2-Thinking) runs on OpenRouter.")
    out_dir = OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = persistbench.load_subset()

    # Generations are saved before judging, so a judge failure doesn't cost a re-generation.
    gen_path = out_dir / "generations.json"
    if gen_path.exists():
        gens = json.loads(gen_path.read_text())
    else:
        llm = llm or policy.load(model)
        outs = policy.generate(llm, [policy.prompt(llm, r["memories"], r["query"]) for r in rows], **config.EVAL_SAMPLING)
        gens = {r["id"]: {**o[0], "text": strip_think(o[0]["text"])} for r, o in zip(rows, outs)}
        gen_path.write_text(json.dumps(gens, indent=2))

    # Reuse verdicts from an earlier run of this label; only items without a score are (re)judged.
    results_path = out_dir / "results.json"
    done = {}
    if results_path.exists():
        done = {r["id"]: {k: r[k] for k in ("score", "reasoning") if k in r}
                for r in json.loads(results_path.read_text())["results"] if r.get("score") is not None}
    todo = [r for r in rows if r["id"] not in done]
    with ThreadPoolExecutor(16) as pool:
        done |= dict(zip((r["id"] for r in todo), pool.map(lambda r: persistbench.judge(r, gens[r["id"]]["text"]), todo)))

    results = [
        {"id": r["id"], "failure_type": r["failure_type"], "query": r["query"], **gens[r["id"]], **done[r["id"]]}
        for r in rows
    ]
    summary = {"label": label, "model": model, **persistbench.summarize(results)}
    results_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    s = summary
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"
    print(f"\nPersistBench — {label} ({model}), one generation per item\n")
    print(f"Cross-domain leakage failure rate:   {pct(s['leakage_failure_rate'])}  (n={s['n_leakage']})")
    print(f"Beneficial failure rate:             {pct(s['beneficial_failure_rate'])}  (n={s['n_beneficial']}; "
          f"not perfect: {pct(s['beneficial_not_perfect_rate'])})")
    print(f"Answers: mean {s['mean_tokens']:.0f} tokens, truncated {pct(s['truncation_rate'])}")
    if s["n_judge_errors"]:
        print(f"Judge errors: {s['n_judge_errors']} (rerun the same command to retry; generations are reused)")
    print(f"\nFull results: {results_path}")
    return summary
