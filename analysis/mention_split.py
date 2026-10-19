"""Memory-mention rate on saved training rollouts, split by relevant/irrelevant prompts (Table 3's data).

Re-judges a fixed sample of rollouts per (arm, step, kind) with the training judge's mentions_memory: training
itself only labeled irrelevant rollouts. Needs OPENAI_API_KEY and the runs' rollouts/ folders.
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor

import config
from training.judge import mentions_memory

ARMS = ["base", "dropout", "fixed"]
STEPS = [0, 50, 100, 150, 200, 250, 290]  # rollouts are logged every 10 steps; 290 is the last
N = 24  # answers per (arm, step, kind)


def run(runs_dir, out, n=N):
    examples = {e["id"]: e for e in map(json.loads, open(config.DATA_DIR / "training_set.jsonl"))}
    jobs = []
    for arm in ARMS:
        for step in STEPS:
            rows = [json.loads(line) for line in open(runs_dir / arm / f"rollouts/step_{step:04d}.jsonl")]
            for relevant in (True, False):
                pool = [r for r in rows if r["relevant"] == relevant]
                # Seeded by step, so all arms sample the same positions in their rollout files.
                for r in random.Random(step).sample(pool, min(n, len(pool))):
                    jobs.append((arm, step, relevant, examples[r["id"]], r["answer"]))

    def one(job):
        arm, step, relevant, ex, answer = job
        return [arm, step, relevant, int(mentions_memory(ex["memories"], ex["query"], answer)[0])]

    with ThreadPoolExecutor(24) as pool:
        results = list(pool.map(one, jobs))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results))
    print(f"Wrote {len(results)} labels to {out}")
