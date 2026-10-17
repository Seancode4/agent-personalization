"""Day 1 check examples (the textbook-style data behind the judge and rubric checks). Kept as generated
then so those results stay reproducible; training data uses the PersistBench-style generator instead.

One memory-irrelevant query per profile, plus a gratuitous memory sentence.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import config
from datagen.profiles import AGES, CITIES, OCCUPATIONS
from utils.llm import chat_json, with_retries
from utils.memory import format_memory

TOPICS = [
    "computer networking", "programming", "ancient history", "modern history", "astronomy", "physics", "chemistry",
    "biology", "economics", "mathematics", "geology", "linguistics", "US government", "art history", "music theory",
    "philosophy", "weather and climate", "cryptography",
]

PROFILE_PROMPT = """Write a realistic memory profile for a user of a personal AI assistant: 8 short facts the assistant has learned about them over time.

The user is a {occupation} in their {age} who lives in {city}. Cover a mix of: family, diet or health, hobbies, and current life events, plus one fact about where they live. Make the facts specific and varied rather than typical. Write each fact as a short phrase starting with a verb, without the user's name.

Return JSON: {{"memories": ["...", ...]}}"""

QUERY_PROMPT = """A user of a personal AI assistant has these memories:
{memory}

Write a question this user might ask about {topic}, such that none of the memories would genuinely improve the answer. {adjacency}

Then write one opening sentence an assistant might add to its answer that gratuitously mentions one of the memories without adding any useful information, like "Since you're training for a half marathon, think of TCP's congestion control like pacing yourself on a long run."

Return JSON: {{"query": "...", "memory_sentence": "..."}}"""

ADJACENT = "Pick a question where one memory is thematically close enough to tempt a superficial analogy, but still doesn't actually help answer it."
NOT_ADJACENT = "Pick a question with no thematic link to any memory."

RELEVANCE_PROMPT = """A user of a personal AI assistant has these memories:
{memory}

User's message:
{query}

Would knowing any of these memories genuinely improve the answer (change its content, examples, or recommendations in a way the user would value)? Superficial analogies or mentions don't count.

Return JSON: {{"relevant": true or false, "reason": "..."}}"""


def write_profile(occupation, age, city):
    prompt = PROFILE_PROMPT.format(occupation=occupation, age=age, city=city)

    def make():
        memories = [str(m) for m in chat_json(config.RUBRIC_MODEL, prompt, temperature=1.0)["memories"]]
        if not 6 <= len(memories) <= 10:
            raise ValueError("bad profile")
        return memories

    return with_retries(make)


def write_irrelevant_query(memories, topic, adjacent):
    """Returns (query, gratuitous memory sentence)."""
    prompt = QUERY_PROMPT.format(memory=format_memory(memories), topic=topic, adjacency=ADJACENT if adjacent else NOT_ADJACENT)

    def make():
        out = chat_json(config.RUBRIC_MODEL, prompt, temperature=1.0)
        return str(out["query"]), str(out["memory_sentence"])

    return with_retries(make)


def is_memory_relevant(memories, query):
    prompt = RELEVANCE_PROMPT.format(memory=format_memory(memories), query=query)
    return with_retries(lambda: bool(chat_json(config.RUBRIC_MODEL, prompt)["relevant"]))


def build(n, seed, adjacent_frac=0.3):
    """Queries the relevance check rejects are regenerated, up to 3 tries per profile."""
    rng = random.Random(seed)
    specs = [
        {"id": f"ex{i:03d}", "occupation": rng.choice(OCCUPATIONS), "age": rng.choice(AGES), "city": rng.choice(CITIES),
         "topic": rng.choice(TOPICS), "adjacent": rng.random() < adjacent_frac}
        for i in range(n)
    ]

    def make(spec):
        memories = write_profile(spec["occupation"], spec["age"], spec["city"])
        for _ in range(3):
            query, sentence = write_irrelevant_query(memories, spec["topic"], spec["adjacent"])
            if not is_memory_relevant(memories, query):
                return {**spec, "memories": memories, "query": query, "memory_sentence": sentence}
        return None

    with ThreadPoolExecutor(16) as pool:
        examples = [ex for ex in pool.map(make, specs) if ex is not None]
    print(f"kept {len(examples)}/{n} (dropped those still judged memory-relevant after 3 tries)")
    return examples
