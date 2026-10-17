"""GRPO training of the policy against the rubric judge, with optional Rubric Dropout.

    uv run python -m training.train --output-dir DIR [--rubric natural|fixed] [--dropout 0.5]
                                    [--personalization-weight K] [--max-steps 300] [--resume] ...

Runs (all on the same data, seed, and settings):
    Base:              --rubric natural
    Dropout:           --rubric natural --dropout 0.5
    Fixed:             --rubric fixed
    Positive control:  --rubric natural --personalization-weight 3 --max-steps 50

Writes to --output-dir: config.json, log.jsonl (one line per step), rollouts/step_XXXX.jsonl (every
--rollout-every steps), and checkpoints/checkpoint-N/ (LoRA adapters, every --save-steps steps).
"""

import argparse
import hashlib
import json
import random
import subprocess
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


def load_examples(path):
    return [json.loads(line) for line in path.read_text().splitlines() if json.loads(line)["split"] == "train"]


def make_reward_fn(args, out_dir):
    """The reward TRL calls once per generation batch, with every dataset column as a keyword argument."""
    mention_history = deque(maxlen=10)
    rollouts_dir = out_dir / "rollouts"

    def rubric_reward(prompts, completions, id, relevant, memories, query, rubric, trainer_state, log_metric, **_):
        step = trainer_state.global_step
        texts = [strip_think(c[0]["content"]) for c in completions]
        examples = [{"rubric": json.loads(r), "relevant": rel} for r, rel in zip(rubric, relevant)]
        rubrics = [run_rubric(ex, args.rubric, args.personalization_weight) for ex in examples]

        # Memory-mention check on a sample of irrelevant answers: the early signal for the hack.
        irrelevant = [i for i in range(len(texts)) if not relevant[i]]
        sampled = set(random.Random(step).sample(irrelevant, min(args.mention_sample, len(irrelevant))))

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

        with ThreadPoolExecutor(args.judge_workers) as pool:
            judged = list(pool.map(judge_one, range(len(texts))))

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
        log_metric("rubric/full_reward", mean(f for f in full if f is not None) if failures < len(full) else 0.0)
        log_metric("rubric/kept_fraction", mean(sum(r["mask"]) / len(r["mask"]) for r in records))
        if personalization_passes:
            log_metric("rubric/personalization_pass_irrelevant", mean(personalization_passes))
        if mentions:
            mention_history.append(mean(mentions))
            log_metric("memory/mention_rate", mention_history[-1])
            log_metric("memory/mention_rate_avg10", mean(mention_history))

        if step % args.rollout_every == 0:
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
        "policy_model": config.POLICY_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "git_commit": commit,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rubric", choices=["natural", "fixed"], default="natural")
    parser.add_argument("--dropout", type=float, default=0.0, help="fraction of criteria dropped per prompt per step")
    parser.add_argument("--personalization-weight", type=float, default=1.0, help="positive control only")
    parser.add_argument("--data", type=Path, default=TRAINING_SET)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--prompts-per-step", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=16, help="completions per forward/backward pass")
    parser.add_argument("--max-completion-length", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--vllm-memory", type=float, default=0.35, help="share of GPU memory vLLM reserves")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--rollout-every", type=int, default=10)
    parser.add_argument("--mention-sample", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --output-dir")
    args = parser.parse_args()

    out_dir = args.output_dir
    if (out_dir / "config.json").exists() and not args.resume:
        raise SystemExit(f"{out_dir} already has a run; pass --resume to continue it, or choose another --output-dir.")
    completions_per_step = args.prompts_per_step * args.num_generations
    if completions_per_step % args.micro_batch:
        raise SystemExit("--micro-batch must divide prompts-per-step x num-generations")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_config(args, out_dir)

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    dataset = Dataset.from_list([
        {"prompt": chat_messages(ex["memories"], ex["query"]), "id": ex["id"], "relevant": ex["relevant"],
         "memories": ex["memories"], "query": ex["query"], "rubric": json.dumps(ex["rubric"])}
        for ex in load_examples(args.data)
    ])
    training_args = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=completions_per_step // args.micro_batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=0.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"dtype": "bfloat16"},
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=config.POLICY_MODEL,
        reward_funcs=[make_reward_fn(args, out_dir)],
        args=training_args,
        train_dataset=dataset,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
        callbacks=[jsonl_logger(out_dir / "log.jsonl")],
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(out_dir / "final"))


if __name__ == "__main__":
    main()
