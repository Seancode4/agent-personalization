import json
from functools import cache

from openai import OpenAI


@cache
def client():
    # Reads the OPENAI_API_KEY env var; keys never live in the repo.
    return OpenAI()


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
