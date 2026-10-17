"""GRPO training of the policy against the rubric judge, with optional Rubric Dropout.

    uv run python -m training.train --output-dir DIR [--rubric natural|fixed] [--dropout 0.5]
                                    [--personalization-weight K] [--max-steps 300] [--resume] ...

Runs (all on the same data, seed, and settings):
    Base:              --rubric natural
    Dropout:           --rubric natural --dropout 0.5
    Fixed:             --rubric fixed
    Positive control:  --rubric natural --personalization-weight 3 --max-steps 50

Everything else is fixed below (SETTINGS), so every run trains identically apart from those flags.

Writes to --output-dir: config.json, log.jsonl (one line per step), rollouts/step_XXXX.jsonl (every
ROLLOUT_EVERY steps), and checkpoints/checkpoint-N/ (LoRA adapters, every SAVE_STEPS steps).
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import config
from training.judge import grade, mentions_memory
from training.reward import dropout_mask, run_rubric, weighted_reward
from utils.llm import strip_think
from utils.policy import chat_messages

TRAINING_SET = config.DATA_DIR / "training_set.jsonl"

# Shared by every run. Settled in the smoke tests (Oct 17): micro-batch 16 runs out of memory on an L40 with vLLM
# colocated; micro-batch 8, vLLM memory 0.5, and vLLM sleep mode gave no speedup; per-token truncated importance
# sampling replaces TRL's default (sequence_mask), which zeroed whole ~400-token answers.
SEED = 0
LEARNING_RATE = 1e-5
LORA_RANK = 16
PROMPTS_PER_STEP = 16
NUM_GENERATIONS = 8
MICRO_BATCH = 4  # completions per forward/backward pass
VLLM_MEMORY = 0.35  # share of GPU memory vLLM reserves
IS_MODE = "token_truncate"
SAVE_STEPS = 25
ROLLOUT_EVERY = 10
MENTION_SAMPLE = 16  # irrelevant answers checked for memory mentions per step
JUDGE_WORKERS = 32
# Stop training if more than this fraction of a step's grading calls fail for this many steps in a row (an API
# outage or a spending limit): better to resume from the last checkpoint than to train on empty rewards.
GUARD_FAIL_FRACTION = 0.5
GUARD_BAD_STEPS = 3
SETTINGS = {k: v for k, v in globals().items() if k.isupper() and k not in ("TRAINING_SET", "SETTINGS")}


def load_examples(path):
    return [json.loads(line) for line in path.read_text().splitlines() if json.loads(line)["split"] == "train"]


def make_reward_fn(args, out_dir):
    """The reward TRL calls once per generation batch, with every dataset column as a keyword argument."""
    mention_history = deque(maxlen=10)
    rollouts_dir = out_dir / "rollouts"
    bad_steps = [0]  # consecutive steps where most grading calls failed

    def rubric_reward(prompts, completions, id, relevant, memories, query, rubric, trainer_state, log_metric, **_):
        step = trainer_state.global_step
        texts = [strip_think(c[0]["content"]) for c in completions]
        examples = [{"rubric": json.loads(r), "relevant": rel} for r, rel in zip(rubric, relevant)]
        rubrics = [run_rubric(ex, args.rubric, args.personalization_weight) for ex in examples]

        # Memory-mention check on a sample of irrelevant answers: the early signal for the hack.
        irrelevant = [i for i in range(len(texts)) if not relevant[i]]
        sampled = set(random.Random(step).sample(irrelevant, min(MENTION_SAMPLE, len(irrelevant))))

        def judge_one(i):
            try:
                verdicts = grade(memories[i], query[i], texts[i], rubrics[i])
            except Exception:
                verdicts = None
            mention = None
            if i in sampled:
                try:
                    mention = mentions_memory(memories[i], query[i], texts[i])[0]
                except Exception:
                    pass
            return verdicts, mention

        start = time.time()
        with ThreadPoolExecutor(JUDGE_WORKERS) as pool:
            judged = list(pool.map(judge_one, range(len(texts))))
        judge_seconds = time.time() - start

        rewards, full, records = [], [], []
        for i, (verdicts, mention) in enumerate(judged):
            n = len(rubrics[i])
            mask = dropout_mask(n, id[i], step, args.dropout) if args.dropout > 0 else [1] * n
            rewards.append(None if verdicts is None else weighted_reward(rubrics[i], verdicts, mask))
            full.append(None if verdicts is None else weighted_reward(rubrics[i], verdicts))
            records.append({"step": step, "id": id[i], "relevant": relevant[i], "answer": texts[i],
                            "verdicts": verdicts, "mask": mask, "reward": rewards[-1], "full_reward": full[-1],
                            "mentions_memory": mention})

        # A failed judge call gets its group's mean reward: no advantage either way. A fully failed group gets 0
        # for every answer, which also gives zero advantage.
        failures = sum(r is None for r in rewards)
        bad_steps[0] = bad_steps[0] + 1 if failures > GUARD_FAIL_FRACTION * len(rewards) else 0
        if bad_steps[0] >= GUARD_BAD_STEPS:
            raise RuntimeError(
                f"Judge failed on more than {GUARD_FAIL_FRACTION:.0%} of answers for {GUARD_BAD_STEPS} steps in a row "
                f"(step {step}): likely an API outage or a spending limit. Fix it, then rerun the same command with "
                f"--resume to continue from the last checkpoint.")
        for i, r in enumerate(rewards):
            if r is None:
                group = [rewards[j] for j in range(len(rewards)) if id[j] == id[i] and rewards[j] is not None]
                rewards[i] = mean(group) if group else 0.0

        personalization_passes = [
            v for i, (verdicts, _) in enumerate(judged) if verdicts is not None and not relevant[i]
            for c, v in zip(rubrics[i], verdicts) if c["personalization"]
        ]
        mentions = [m for _, m in judged if m is not None]
        log_metric("judge/failures", failures)
        log_metric("time/judge_seconds", judge_seconds)
        log_metric("rubric/full_reward", mean(f for f in full if f is not None) if failures < len(full) else 0.0)
        log_metric("rubric/kept_fraction", mean(sum(r["mask"]) / len(r["mask"]) for r in records))
        if personalization_passes:
            log_metric("rubric/personalization_pass_irrelevant", mean(personalization_passes))
        if mentions:
            mention_history.append(mean(mentions))
            log_metric("memory/mention_rate", mention_history[-1])
            log_metric("memory/mention_rate_avg10", mean(mention_history))

        if step % ROLLOUT_EVERY == 0:
            rollouts_dir.mkdir(parents=True, exist_ok=True)
            with open(rollouts_dir / f"step_{step:04d}.jsonl", "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)
        return rewards

    return rubric_reward


def jsonl_logger(path):
    from transformers import TrainerCallback

    class JsonlLogger(TrainerCallback):
        """Appends every logged metric dict (TRL's and ours) to log.jsonl, one line per logging step."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                with open(path, "a") as f:
                    f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    return JsonlLogger()


def write_config(args, out_dir):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    except OSError:
        commit = None
    (out_dir / "config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        **{k.lower(): v for k, v in SETTINGS.items()},
        "max_completion_length": config.MAX_NEW_TOKENS,
        "policy_model": config.POLICY_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "git_commit": commit,
        "data_sha256": hashlib.sha256(TRAINING_SET.read_bytes()).hexdigest(),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rubric", choices=["natural", "fixed"], default="natural")
    parser.add_argument("--dropout", type=float, default=0.0, help="fraction of criteria dropped per prompt per step")
    parser.add_argument("--personalization-weight", type=float, default=1.0, help="positive control only")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --output-dir")
    args = parser.parse_args()

    out_dir = args.output_dir
    if (out_dir / "config.json").exists() and not args.resume:
        raise SystemExit(f"{out_dir} already has a run; pass --resume to continue it, or choose another --output-dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_config(args, out_dir)

    # Less fragmentation in PyTorch's GPU allocator: fewer out-of-memory errors with vLLM colocated. Must be set
    # before torch is imported.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    dataset = Dataset.from_list([
        {"prompt": chat_messages(ex["memories"], ex["query"]), "id": ex["id"], "relevant": ex["relevant"],
         "memories": ex["memories"], "query": ex["query"], "rubric": json.dumps(ex["rubric"])}
        for ex in load_examples(TRAINING_SET)
    ])
    training_args = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=args.max_steps,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=PROMPTS_PER_STEP * NUM_GENERATIONS // MICRO_BATCH,
        num_generations=NUM_GENERATIONS,
        max_completion_length=config.MAX_NEW_TOKENS,
        temperature=1.0,
        beta=0.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=VLLM_MEMORY,
        vllm_importance_sampling_mode=IS_MODE,
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"dtype": "bfloat16"},
        logging_steps=1,
        save_steps=SAVE_STEPS,
        report_to="none",
        seed=SEED,
    )
    trainer = GRPOTrainer(
        model=config.POLICY_MODEL,
        reward_funcs=[make_reward_fn(args, out_dir)],
        args=training_args,
        train_dataset=dataset,
        peft_config=LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
        callbacks=[jsonl_logger(out_dir / "log.jsonl")],
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(out_dir / "final"))


if __name__ == "__main__":
    main()
