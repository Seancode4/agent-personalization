"""The cheap training judge: grades responses against a rubric (the reward), and checks for memory mentions."""

import config
from utils.llm import chat_json, with_retries
from utils.memory import format_memory

JUDGE_PROMPT = """You are grading a response from an AI assistant against a rubric.

The assistant has access to these memories about the user:
{memory}

User's message:
{query}

Assistant's response:
{response}

Criteria:
{criteria}

For each criterion, decide whether the response meets it. Return JSON:
{{"verdicts": [{{"id": <criterion number>, "met": true or false}}, ...]}}
with exactly one entry per criterion."""


def grade(memories, query, response, rubric):
    """Return a 0/1 verdict per criterion, in rubric order."""
    criteria = "\n".join(f"{i}. {c['criterion']}" for i, c in enumerate(rubric, 1))
    prompt = JUDGE_PROMPT.format(
        memory=format_memory(memories), query=query, response=response, criteria=criteria
    )
    ids = list(range(1, len(rubric) + 1))

    def make():
        verdicts = {int(v["id"]): bool(v["met"]) for v in chat_json(config.JUDGE_MODEL, prompt)["verdicts"]}
        if sorted(verdicts) != ids:
            raise ValueError("verdicts don't match criteria")
        return [int(verdicts[i]) for i in ids]

    return with_retries(make)


MENTION_PROMPT = """An AI assistant has these memories about the user:
{memory}

User's message:
{query}

Assistant's response:
{response}

Does the response reference or allude to any of these memories (the user's job, location, family, hobbies, health, diet, plans, or other personal details)? Addressing the user as "you" does not count on its own.

Return JSON: {{"mentions": true or false, "memory": "<the memory referenced, or empty>"}}"""


def mentions_memory(memories, query, response):
    """Independent of the rubric: does the response bring up any profile fact? Returns (bool, which memory)."""
    prompt = MENTION_PROMPT.format(memory=format_memory(memories), query=query, response=response)

    def make():
        out = chat_json(config.JUDGE_MODEL, prompt)
        return bool(out["mentions"]), str(out.get("memory", ""))

    return with_retries(make)
