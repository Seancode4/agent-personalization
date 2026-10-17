"""Day 1 checks, run before any training.

    uv run python -m checks judge   [--data PATH] [--repeats N] [--regenerate]
    uv run python -m checks rubrics [--data PATH] [--regenerate]
    uv run python -m checks env                                       # GPU
    uv run python -m checks policy  [--data PATH] [--prompts N] [--samples N]   # GPU
    uv run python -m checks persist                                   # GPU, needs OPENROUTER_API_KEY

judge:   does the training judge pay for name-dropping?
rubrics: how often do generated rubrics include a personalization criterion?
env:     GPU, caches, keys, and vLLM are set up.
policy:  the untrained policy's memory mentions, reward spread, and length, on irrelevant and relevant prompts.
persist: step-0 PersistBench on the untrained policy.

judge, rubrics and policy share cached answers and rubrics under outputs/checks/<data file stem>/.
"""

import argparse
from pathlib import Path

from checks.common import DEFAULT_DATA, TRAINING_SET


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
    p.add_argument("--data", type=Path, default=TRAINING_SET, help="needs both relevant and irrelevant examples")
    p.add_argument("--prompts", type=int, default=20, help="per kind (irrelevant and relevant)")
    p.add_argument("--samples", type=int, default=8)
    sub.add_parser("persist", help="step-0 PersistBench on the untrained policy (GPU)")
    args = parser.parse_args()

    # Import per check, so the API-only checks don't load GPU code.
    if args.check == "judge":
        from checks import judge_check

        judge_check.run(args.data, args.repeats, args.regenerate)
    elif args.check == "rubrics":
        from checks import rubric_check

        rubric_check.run(args.data, args.regenerate)
    elif args.check == "env":
        from checks import env_check

        env_check.run()
    elif args.check == "policy":
        from checks import policy_check

        policy_check.run(args.data, args.prompts, args.samples)
    elif args.check == "persist":
        from checks import persist_check

        persist_check.run()


if __name__ == "__main__":
    main()
