"""Tables 1-3, recomputed from per-item results (no GPU, no API calls).

    uv run python -m analysis

Table 1: PersistBench cross-domain leakage failure rate per arm and checkpoint.
Table 2: PersistBench beneficial failure rate (PersistBench's code: rating 1 fails), same checkpoints.
  Both are recounted from every item's judge score in outputs/persistbench/<label>/results.json, with
  PersistBench's scoring (evaluation/persistbench.summarize); the stored summaries are not used.
Table 3: memory-mention rate on training rollouts, irrelevant and relevant prompts, from every label in
  outputs/analysis/mention_split.json (written by analysis/mention_split.py).

Natural rate: share of rubric criteria on memory-irrelevant prompts labeled as personalization criteria, from
  data/training_set.jsonl (all splits).

Prints the tables as CSV and writes them to outputs/analysis/table{1,2,3}_*.csv; prints the natural rate.
"""

import argparse
import csv
import io
import json
from statistics import mean

import config
from evaluation import evaluate, persistbench

ARMS = ["base", "dropout", "fixed"]
STEPS = [50, 100, 200, 300]  # PersistBench checkpoints; step 0 is the untrained model, shared by all arms
LATE = (200, 300)  # late window: mean of the PersistBench checkpoints in this range
MENTIONS = config.OUTPUTS_DIR / "analysis/mention_split.json"
OUT = config.OUTPUTS_DIR / "analysis"


def persist_rates():
    """{arm: {step: summary}}, recomputed from each item's judge score."""
    summary = lambda label: persistbench.summarize(json.loads((evaluate.OUT / label / "results.json").read_text())["results"])
    step0 = summary("step0")
    return {arm: {0: step0, **{step: summary(f"{arm}-step{step}") for step in STEPS}} for arm in ARMS}


def mention_rates():
    """{(arm, relevant): {step: rate}} from mention_split's labels."""
    rates = {}
    for arm, step, relevant, mention in json.loads(MENTIONS.read_text()):
        rates.setdefault((arm, relevant), {}).setdefault(step, []).append(mention)
    return {k: {s: mean(m) for s, m in v.items()} for k, v in rates.items()}


def write_csv(name, header, rows):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows([header] + rows)
    (OUT / name).write_text(buf.getvalue())
    print(f"{name}\n{buf.getvalue()}")


def persist_table(rates, key, name, late=False):
    steps = [0] + STEPS
    rows = []
    for arm in ARMS:
        row = [arm] + [round(rates[arm][s][key], 2) for s in steps]
        if late:
            row.append(round(mean(rates[arm][s][key] for s in steps if LATE[0] <= s <= LATE[1]), 2))
        rows.append(row)
    write_csv(name, ["arm"] + steps + ([f"{LATE[0]}-{LATE[1]}"] if late else []), rows)


def mention_table(rates, name):
    steps = sorted(next(iter(rates.values())))
    rows = [[arm, "relevant" if rel else "irrelevant"] + [round(rates[(arm, rel)][s], 2) for s in steps]
            for arm in ARMS for rel in (False, True)]
    write_csv(name, ["arm", "prompt_type"] + steps, rows)


def natural_rate():
    """Share of rubric criteria on memory-irrelevant prompts labeled as personalization criteria."""
    examples = [json.loads(line) for line in open(config.DATA_DIR / "training_set.jsonl")]
    criteria = [c for e in examples if not e["relevant"] for c in e["rubric"]]
    n = sum(c["personalization"] for c in criteria)
    print(f"Natural rate: {n / len(criteria):.0%} of criteria on irrelevant prompts ({n} of {len(criteria)}) "
          f"are personalization criteria")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rates = persist_rates()
    persist_table(rates, "leakage_failure_rate", "table1_leakage.csv", late=True)
    persist_table(rates, "beneficial_failure_rate", "table2_beneficial.csv")
    mention_table(mention_rates(), "table3_mentions.csv")
    natural_rate()


if __name__ == "__main__":
    main()
