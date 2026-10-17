import config
from llm import chat_json
from memory import format_memory

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


def grade(memories, query, response, rubric, tries=3):
    """Return a 0/1 verdict per criterion, in rubric order."""
    criteria = "\n".join(f"{i}. {c['criterion']}" for i, c in enumerate(rubric, 1))
    prompt = JUDGE_PROMPT.format(
        memory=format_memory(memories), query=query, response=response, criteria=criteria
    )
    ids = list(range(1, len(rubric) + 1))
    for _ in range(tries):
        out = chat_json(config.JUDGE_MODEL, prompt)
        try:
            verdicts = {int(v["id"]): bool(v["met"]) for v in out["verdicts"]}
        except (KeyError, TypeError, ValueError):
            continue
        if sorted(verdicts) == ids:
            return [int(verdicts[i]) for i in ids]
    raise ValueError(f"judge returned malformed verdicts after {tries} tries")
