"""The situation bank: distinct real-life situations per domain, so queries don't keep reusing the same few ideas.
Built once, skimmed by hand, then drawn from without replacement."""

import json
from concurrent.futures import ThreadPoolExecutor

import config
from utils.llm import chat_json, with_retries

# PersistBench's nine memory/query domains (memory_domain / query_domain in its dataset).
DOMAINS = [
    "Educational and Formative Experiences",
    "Financial and Legal Matters",
    "Health and Medical Information",
    "Intimate and Romantic Relationships",
    "Personal Beliefs (Political, Religious, and Social)",
    "Private Thoughts and Journals",
    "Professional and Work Life",
    "Self-Concept and Identity",
    "Social and Relational Information",
]
BANK = config.DATA_DIR / "situation_bank.json"

SITUATIONS_PROMPT = """List {n} distinct, specific situations in which a person might ask a personal AI assistant for help, in the domain: {domain}.

Each is one line describing a situation and what is needed, not a question. Do not say who the person is (no ages, jobs, genders, or life stages like "a student" or "a retiree"): each situation must fit any adult, because it will be paired with an unrelated user profile. Start each with a verb or noun phrase, e.g. "Deciding whether..." or "Conflict with...", not "A person...". Each must be something happening in the person's own life right now, with a concrete trigger and real stakes (a decision with a deadline, a conflict, a setback, an awkward conversation), not research, reflection, or general self-improvement. Vary them widely: everyday and serious, practical and emotional. Make each concrete enough that two situations never describe the same problem.{avoid}

Return JSON: {{"situations": ["...", ...]}}"""


def write_situations(domain, per_domain=120, batch=40):
    """In batches, telling each batch what earlier batches already covered."""
    situations = []
    while len(situations) < per_domain:
        avoid = ("\n\nAlready listed (do not repeat or closely paraphrase):\n" + "\n".join(f"- {s}" for s in situations)
                 if situations else "")
        prompt = SITUATIONS_PROMPT.format(n=batch, domain=domain, avoid=avoid)

        def make():
            out = [str(s).strip() for s in chat_json(config.RUBRIC_MODEL, prompt, temperature=1.0)["situations"]]
            if len(out) < batch // 2:
                raise ValueError("too few situations")
            return out

        situations += with_retries(make)
    return situations[:per_domain]


def build_bank(per_domain=120):
    with ThreadPoolExecutor(len(DOMAINS)) as pool:
        bank = dict(zip(DOMAINS, pool.map(lambda d: write_situations(d, per_domain), DOMAINS)))
    BANK.write_text(json.dumps(bank, indent=2))
    return bank


def load_bank():
    if not BANK.exists():
        raise SystemExit(f"No situation bank at {BANK}; run `uv run python -m datagen situations` first.")
    return json.loads(BANK.read_text())
