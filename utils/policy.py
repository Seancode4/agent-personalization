"""vLLM generation for the policy. Imports vllm lazily so the rest of the repo works without a GPU."""

import json
from dataclasses import dataclass
from pathlib import Path

import config
from utils.memory import policy_system_prompt


@dataclass
class Policy:
    llm: object
    lora: object = None  # vllm LoRARequest when serving a training checkpoint


def load(model=config.POLICY_MODEL, gpu_memory_utilization=0.85, max_model_len=8192):
    """A Hugging Face id or local model path, or a LoRA checkpoint directory from training."""
    from vllm import LLM

    adapter = Path(model) / "adapter_config.json"
    if adapter.exists():
        from vllm.lora.request import LoRARequest

        adapter_config = json.loads(adapter.read_text())
        max_rank = max(8, 1 << (adapter_config["r"] - 1).bit_length())  # vLLM wants a power of two
        llm = LLM(model=adapter_config.get("base_model_name_or_path") or config.POLICY_MODEL, enable_lora=True,
                  max_lora_rank=max_rank, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
        return Policy(llm, LoRARequest("checkpoint", 1, str(model)))
    return Policy(LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len))


def chat_messages(memories, query):
    """The policy's input: the same in training (TRL rollouts) and evaluation."""
    return [{"role": "system", "content": policy_system_prompt(memories)}, {"role": "user", "content": query}]


def prompt(policy, memories, query):
    """Chat-formatted prompt with Qwen3's thinking mode off."""
    return policy.llm.get_tokenizer().apply_chat_template(
        chat_messages(memories, query), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate(policy, prompts, n=1, temperature=1.0, top_p=1.0, max_tokens=config.MAX_NEW_TOKENS):
    """Returns, per prompt, a list of n {text, n_tokens, truncated}."""
    from vllm import SamplingParams

    params = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    return [
        [{"text": o.text, "n_tokens": len(o.token_ids), "truncated": o.finish_reason == "length"} for o in out.outputs]
        for out in policy.llm.generate(prompts, params, lora_request=policy.lora)
    ]
