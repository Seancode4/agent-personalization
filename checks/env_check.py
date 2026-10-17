"""Environment check (GPU): torch sees the GPU, caches are off the home directory, keys are set,
and vLLM loads the policy and generates with thinking mode off.
"""

import json
import os
import time
from pathlib import Path

from checks.common import CHECK_EXAMPLES
from utils import policy


def run():
    import torch

    ok = True
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} / {total / 2**30:.1f} GB free")
    else:
        ok = False
    home = str(Path.home())
    for var in ("HF_HOME", "UV_CACHE_DIR", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME"):
        value = os.environ.get(var, "")
        bad = not value or value.startswith(home)
        ok &= not bad
        print(f"{var:<16}{value or '(unset)'}{'   <- should point off the home directory' if bad else ''}")
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        print(f"{var:<20}{'set' if os.environ.get(var) else 'MISSING'}")
    if not ok:
        raise SystemExit("\nFix the items above before loading the model.")


    examples = json.loads(CHECK_EXAMPLES.read_text())[:2]
    llm = policy.load()
    start = time.time()
    outs = policy.generate(llm, [policy.prompt(llm, ex["memories"], ex["query"]) for ex in examples], n=4)
    elapsed = time.time() - start
    tokens = sum(o["n_tokens"] for group in outs for o in group)
    thinking = any("<think>" in o["text"] for group in outs for o in group)
    print(f"\nGenerated 8 answers, {tokens} tokens in {elapsed:.1f}s ({tokens / elapsed:.0f} tokens/s)")
    print(f"Thinking tags present: {thinking}{'   <- thinking mode should be off' if thinking else ''}")
    print(f"\nQ: {examples[0]['query']}\nA: {outs[0][0]['text'][:600]}")
