"""Data generation.

    uv run python -m datagen situations      [--per-domain 120] [--force]
    uv run python -m datagen training-set    [--profiles 250] [--queries-per-profile 5] [--seed 0]
    uv run python -m datagen report          [--data PATH]
    uv run python -m datagen check-examples  [--n 50] [--seed 0]

situations:     build the situation bank (data/situation_bank.json). Skim it before generating.
training-set:   PersistBench-style training data (data/training_set.jsonl), then print the report.
report:         reprint the report for an existing training set.
check-examples: the textbook-style examples behind the Day 1 judge and rubric checks.
"""

import argparse
import json
from pathlib import Path

import config
from datagen import check_examples, situations, training_set

TRAINING_SET = config.DATA_DIR / "training_set.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("situations", help="build the situation bank")
    p.add_argument("--per-domain", type=int, default=120)
    p.add_argument("--force", action="store_true", help="overwrite an existing bank")
    p = sub.add_parser("training-set", help="generate the training set")
    p.add_argument("--profiles", type=int, default=250)
    p.add_argument("--queries-per-profile", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=TRAINING_SET)
    p = sub.add_parser("report", help="reprint the report for a training set")
    p.add_argument("--data", type=Path, default=TRAINING_SET)
    p = sub.add_parser("check-examples", help="examples for the Day 1 checks")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=config.DATA_DIR / "check_examples.json")
    args = parser.parse_args()

    if args.cmd == "situations":
        if situations.BANK.exists() and not args.force:
            raise SystemExit(f"{situations.BANK} exists; pass --force to rebuild it.")
        bank = situations.build_bank(args.per_domain)
        print(f"wrote {situations.BANK}: " + ", ".join(f"{d.split(' ')[0]} {len(s)}" for d, s in bank.items()))
    elif args.cmd == "training-set":
        examples, log, refills = training_set.build(args.profiles, args.queries_per_profile, args.seed)
        args.out.write_text("".join(json.dumps(ex) + "\n" for ex in examples))
        training_set.report(examples, log, refills)
        print(f"\nwrote {args.out}")
    elif args.cmd == "report":
        training_set.report([json.loads(line) for line in args.data.read_text().splitlines()])
    elif args.cmd == "check-examples":
        examples = check_examples.build(args.n, args.seed)
        args.out.write_text(json.dumps(examples, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
