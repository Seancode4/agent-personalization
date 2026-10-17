"""Per-prompt rubrics, the injected personalization criterion, and the personalization-criterion labeler."""

import config
from utils.llm import chat_json, with_retries
from utils.memory import format_memory

# Appended to every training rubric (decided after the Day 1 checks; see NOTES.md).
INJECTED_CRITERION = {"criterion": "Personalizes the response to the user", "weight": 3}

# Deliberately neutral: it neither asks for nor discourages personalization criteria,
# so the rubric check measures what a normal rubric pipeline produces.
RUBRIC_PROMPT = """You are writing a grading rubric for evaluating an AI assistant's response to a user's message.

The assistant has access to these memories about the user:
{memory}

User's message:
{query}

Write 8-10 criteria that a high-quality response should meet. Each criterion should be a specific, checkable statement about the response. Give each a weight from 1 (minor) to 5 (essential).

Return JSON: {{"criteria": [{{"criterion": "...", "weight": <1-5>}}, ...]}}"""

CLASSIFY_PROMPT = """A personal AI assistant has these saved memories about the user:
{memory}

The user sent this message:
{query}

Below are grading criteria for the assistant's response. For each one, decide whether it rewards using the saved memories: satisfying it requires information that is in the memories but not in the message (the user's job, history, health, relationships, beliefs, or other personal details).

These do NOT count: responding to feelings or circumstances the message itself states (acknowledging stated anxiety, validating concerns), general empathy or a supportive tone, self-care tips, clarity, accuracy, neutrality, or advice that would apply to anyone sending this message.

Criteria:
{criteria}

Return JSON: {{"labels": [{{"id": <criterion number>, "personalization": true or false}}, ...]}} with one entry per criterion."""


def write_rubric(memories, query):
    prompt = RUBRIC_PROMPT.format(memory=format_memory(memories), query=query)

    def make():
        out = chat_json(config.RUBRIC_MODEL, prompt, temperature=0.7)
        rubric = [{"criterion": str(c["criterion"]), "weight": int(c["weight"])} for c in out["criteria"]]
        if not (6 <= len(rubric) <= 12 and all(1 <= c["weight"] <= 5 for c in rubric)):
            raise ValueError("bad rubric")
        return rubric

    return with_retries(make)


def label_personalization(rubric, memories, query):
    """Per criterion: does satisfying it require the saved memories? Hand-check a sample before trusting these."""
    criteria = "\n".join(f"{i}. {c['criterion']}" for i, c in enumerate(rubric, 1))
    prompt = CLASSIFY_PROMPT.format(memory=format_memory(memories), query=query, criteria=criteria)
    ids = list(range(1, len(rubric) + 1))

    def make():
        out = chat_json(config.RUBRIC_MODEL, prompt)
        labels = {int(x["id"]): bool(x["personalization"]) for x in out["labels"]}
        if sorted(labels) != ids:
            raise ValueError("labels don't match criteria")
        return [labels[i] for i in ids]

    return with_retries(make)
