"""Shared by the checks: example loading and the cached rubrics they all grade against."""

import json

import config
from datagen.rubrics import write_rubric
from utils.cache import cached

OUT = config.OUTPUTS_DIR / "checks"
DEFAULT_DATA = config.DATA_DIR / "judge_check.json"
CHECK_EXAMPLES = config.DATA_DIR / "check_examples.json"
TRAINING_SET = config.DATA_DIR / "training_set.jsonl"


def load(data_path, relevant=False):
    """Memory-irrelevant (or, with relevant=True, memory-relevant) examples from a check-examples JSON file or
    the training set (JSONL), plus the output folder for this dataset. Held-out examples come first."""
    text = data_path.read_text()
    examples = [json.loads(line) for line in text.splitlines()] if data_path.suffix == ".jsonl" else json.loads(text)
    examples = [ex for ex in examples if ex.get("relevant", False) == relevant]
    examples.sort(key=lambda ex: ex.get("split") != "held_out")  # stable: held-out first, original order otherwise
    out_dir = OUT / data_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return examples, out_dir


def get_rubrics(examples, out_dir, regenerate=False):
    """Training-set examples carry their rubric; others get one written once and cached."""
    stored = {ex["id"]: [{"criterion": c["criterion"], "weight": c["weight"]} for c in ex["rubric"]]
              for ex in examples if "rubric" in ex}
    missing = [ex for ex in examples if ex["id"] not in stored]
    if not missing:
        return stored
    return stored | cached(out_dir / "rubrics.json", missing, lambda ex: write_rubric(ex["memories"], ex["query"]), regenerate)
