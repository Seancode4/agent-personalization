"""vLLM generation for the policy. Imports vllm lazily so the rest of the repo works without a GPU."""

import config
from utils.memory import policy_system_prompt


def load(model=config.POLICY_MODEL, gpu_memory_utilization=0.85, max_model_len=8192):
    from vllm import LLM

    return LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)


def prompt(llm, memories, query):
    """Chat-formatted prompt with Qwen3's thinking mode off."""
    messages = [{"role": "system", "content": policy_system_prompt(memories)}, {"role": "user", "content": query}]
    return llm.get_tokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate(llm, prompts, n=1, temperature=1.0, top_p=1.0, max_tokens=config.MAX_NEW_TOKENS):
    """Returns, per prompt, a list of n {text, n_tokens, truncated}."""
    from vllm import SamplingParams

    params = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    return [
        [{"text": o.text, "n_tokens": len(o.token_ids), "truncated": o.finish_reason == "length"} for o in out.outputs]
        for out in llm.generate(prompts, params)
    ]
