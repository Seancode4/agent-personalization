from pathlib import Path

# API keys come from env vars (OPENAI_API_KEY, OPENROUTER_API_KEY), never from this file.

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"  # data only (JSON/JSONL); code lives in the section folders
OUTPUTS_DIR = ROOT / "outputs"

# Check these models are still served before long runs.
RUBRIC_MODEL = "gpt-4o-mini"  # writes rubrics
JUDGE_MODEL = "gpt-4o-mini"  # cheap training judge
ANSWER_MODEL = "gpt-4o-mini"  # writes clean answers for the judge check

# Policy (the model being trained). POLICY_NAME is what the policy is called in its system prompt
# and what vLLM serves it as; it must be identical in training and PersistBench evaluation.
POLICY_MODEL = "Qwen/Qwen3-1.7B"
POLICY_NAME = "Qwen3-1.7B"
MAX_NEW_TOKENS = 1024
# Qwen3's recommended non-thinking sampling, used for evaluation. Training rollouts use temperature 1.0.
EVAL_SAMPLING = {"temperature": 0.7, "top_p": 0.8}

# Vendored PersistBench prompts and eval subset (see third_party/persistbench/README.md).
PERSISTBENCH_DIR = ROOT / "third_party/persistbench"
