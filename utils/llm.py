"""LLM API plumbing: clients, retries, and JSON parsing, shared by every section."""

import json
import os
import re
from functools import cache

from openai import OpenAI

import config

# Errors that mean "the model returned something malformed": worth asking again.
PARSE_ERRORS = (KeyError, TypeError, ValueError)


@cache
def client():
    # OPENAI_API_KEY env var, or config.OPENAI_API_KEY if set there (never commit a key).
    # max_retries: the SDK backs off and retries rate limits (429s) and transient errors itself.
    return OpenAI(api_key=getattr(config, "OPENAI_API_KEY", None) or None, max_retries=8)


@cache
def openrouter():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=8)


def chat(model, prompt, temperature=0.7):
    resp = client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def chat_json(model, prompt, temperature=0.0):
    resp = client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def with_retries(make, tries=3, errors=PARSE_ERRORS):
    """Call make() until it returns instead of raising one of `errors`."""
    for attempt in range(tries):
        try:
            return make()
        except errors as e:
            if attempt == tries - 1:
                raise RuntimeError(f"failed after {tries} tries: {type(e).__name__}: {e}") from e


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(content):
    """Parse JSON from free-form model output: as-is, then an embedded object, then a code block.
    Same fallbacks as PersistBench's utils.extract_json_from_response."""
    candidates = [content]
    if m := re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL):
        candidates.append(m.group(0))
    if m := re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL):
        candidates.append(m.group(1))
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON in response: {content[:200]!r}")
