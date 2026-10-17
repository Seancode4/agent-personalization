"""PersistBench evaluation of the policy at a checkpoint.

    uv run python -m evaluation --label step0 [--model HF_ID_OR_PATH]

Generates one answer per item of the fixed PersistBench subset (100 cross-domain leakage items + all
100 beneficial items) with vLLM, then grades them with PersistBench's judge: its prompts, called the
way PersistBench calls it (see third_party/persistbench/README.md). Needs OPENROUTER_API_KEY.
"""

import argparse

import config
from evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="output folder name, e.g. step0 or base-step150")
    parser.add_argument("--model", default=config.POLICY_MODEL, help="HF id or local path of a (merged) checkpoint")
    args = parser.parse_args()
    evaluate.run(args.label, args.model)


if __name__ == "__main__":
    main()
