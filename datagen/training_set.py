"""The training set: PersistBench-style profiles, each with ~5 first-person queries (half memory-relevant,
half cross-domain irrelevant), gated for relevance, with a rubric and personalization labels per query.

The injected personalization criterion is NOT stored: training/train.py appends it per --rubric mode,
so one dataset serves all three arms.
"""

import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from datagen.profiles import AGES, CITIES, OCCUPATIONS, write_profile
from datagen.queries import TASK_TYPES, gate, write_name_drop, write_query
from datagen.rubrics import label_personalization, write_rubric
from datagen.situations import DOMAINS, load_bank


def plan(n_profiles, queries_per_profile, seed, held_out_frac=0.2):
    """All random choices up front, in code: personas, domains, and situations drawn without replacement."""
    rng = random.Random(seed)
    bank = load_bank()
    pools = {d: rng.sample(s, len(s)) for d, s in bank.items()}
    refills = Counter()

    def draw(domain):
        if not pools[domain]:  # bank exhausted for this domain: start a fresh shuffled pass
            pools[domain] = rng.sample(bank[domain], len(bank[domain]))
            refills[domain] += 1
        return pools[domain].pop()

    profiles = []
    for i in range(n_profiles):
        domains = rng.sample(DOMAINS, rng.randint(3, 5))
        queries = []
        for j in range(queries_per_profile):
            relevant = (i + j) % 2 == 0  # alternates per profile, so the set is ~50/50
            # Relevant: a domain the memories cover. Irrelevant: one they don't (PersistBench's cross-domain setup).
            query_domain = rng.choice(domains if relevant else [d for d in DOMAINS if d not in domains])
            queries.append({
                "id": f"p{i:03d}-q{j}", "relevant": relevant, "query_domain": query_domain,
                "situation": draw(query_domain), "task": rng.choice(TASK_TYPES),
            })
        profiles.append({
            "profile_id": f"p{i:03d}", "occupation": rng.choice(OCCUPATIONS), "age": rng.choice(AGES),
            "city": rng.choice(CITIES), "domains": domains, "n_memories": rng.randint(4, 16), "queries": queries,
        })
    held_out = set(rng.sample([p["profile_id"] for p in profiles], round(held_out_frac * n_profiles)))
    for p in profiles:
        p["split"] = "held_out" if p["profile_id"] in held_out else "train"
    return profiles, refills


def make_profile_examples(p, tries=5):
    """Returns (examples, log). A query passes the gate if it states no memory and its material memories fit its
    label (none for irrelevant, at least one for relevant); failures are rewritten up to `tries` times, then dropped."""
    log = Counter()
    try:
        memories, memory_domains = write_profile(p["occupation"], p["age"], p["city"], p["domains"], p["n_memories"])
    except RuntimeError:
        log["profile_failed"] += 1
        return [], log

    examples = []
    for q in p["queries"]:
        kind = "relevant" if q["relevant"] else "irrelevant"
        for _ in range(tries):
            try:
                query = write_query(memories, q["relevant"], q["situation"], q["task"])
                material, stated = gate(memories, query)
            except RuntimeError:
                continue
            if stated:
                log[f"{kind}_rejected_states_memory"] += 1
            elif (not material) if q["relevant"] else material:
                log[f"{kind}_rejected_relevance"] += 1
            else:
                break
        else:
            log[f"{kind}_dropped"] += 1
            continue
        try:
            rubric = write_rubric(memories, query)
            labels = label_personalization(rubric, memories, query)
            sentence = None if q["relevant"] else write_name_drop(memories, query)
        except RuntimeError:
            log[f"{kind}_dropped"] += 1
            continue
        examples.append({
            "id": q["id"], "profile_id": p["profile_id"], "split": p["split"],
            "memories": memories, "memory_domains": memory_domains,
            "query": query, "query_domain": q["query_domain"], "relevant": q["relevant"],
            "task": q["task"], "situation": q["situation"],
            "material_memories": material,
            "memory_sentence": sentence,  # irrelevant queries only: for rerunning the judge check
            "rubric": [{**c, "personalization": lab} for c, lab in zip(rubric, labels)],
        })
        log[f"{kind}_kept"] += 1
    return examples, log


def build(n_profiles, queries_per_profile, seed):
    profiles, refills = plan(n_profiles, queries_per_profile, seed)

    def safe(p):
        # An API failure that outlasts the SDK's retries loses one profile, not the whole run.
        try:
            return make_profile_examples(p)
        except Exception as e:
            print(f"profile {p['profile_id']} failed: {type(e).__name__}: {e}")
            return [], Counter(profile_failed=1)

    with ThreadPoolExecutor(16) as pool:
        results = []
        for i, r in enumerate(pool.map(safe, profiles), 1):
            results.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(profiles)} profiles done", flush=True)
    examples = [ex for exs, _ in results for ex in exs]
    log = sum((lg for _, lg in results), Counter())
    return examples, log, refills


def opening(text, n=3):
    return " ".join(re.findall(r"[\w']+", text)[:n])


def report(examples, log=None, refills=None, n_samples=20, seed=0):
    """What to look at before training: counts, style vs PersistBench, repetition, and random samples."""
    irrelevant = [ex for ex in examples if not ex["relevant"]]
    profiles = {ex["profile_id"]: ex["memories"] for ex in examples}
    mems = [m for ms in profiles.values() for m in ms]

    print(f"\n{len(examples)} examples from {len(profiles)} profiles: "
          f"{len(examples) - len(irrelevant)} relevant, {len(irrelevant)} irrelevant; "
          f"{sum(ex['split'] == 'train' for ex in examples)} train, {sum(ex['split'] == 'held_out' for ex in examples)} held out")
    if log:
        print("generation log: " + ", ".join(f"{k} {v}" for k, v in sorted(log.items())))
    if refills:
        print("situation bank ran out (reused) in: " + ", ".join(f"{d} x{n}" for d, n in refills.items()))

    print("\nStyle (PersistBench eval subset in brackets):")
    print(f"  memories per profile  {mean(len(m) for m in profiles.values()):.1f}  "
          f"(min {min(len(m) for m in profiles.values())}, max {max(len(m) for m in profiles.values())})   [10.7, 4-16]")
    print(f"  words per memory      {mean(len(m.split()) for m in mems):.1f}   [14.7]")
    print(f"  words per query       {mean(len(ex['query'].split()) for ex in examples):.1f}   [35.9]")
    print(f"  irrelevant rubrics with a natural personalization criterion: "
          f"{mean(any(c['personalization'] for c in ex['rubric']) for ex in irrelevant):.0%}" if irrelevant else "")

    print("\nMost common query openings:")
    for text, n in Counter(opening(ex["query"]) for ex in examples).most_common(8):
        print(f"  {n:>4}  {text}")
    print("Most common memory openings (after 'User'):")
    for text, n in Counter(opening(m.removeprefix("User").strip()) for m in mems).most_common(8):
        print(f"  {n:>4}  {text}")
    dupes = sum(n - 1 for n in Counter(m.lower() for m in mems).values() if n > 1)
    print(f"Exact duplicate memories across profiles: {dupes}")

    print(f"\n{n_samples} random examples:")
    for ex in random.Random(seed).sample(examples, min(n_samples, len(examples))):
        tag = "relevant" if ex["relevant"] else "irrelevant"
        print(f"\n  [{ex['id']}, {tag}, {ex['query_domain']}]")
        print(f"    memory: {ex['memories'][0][:140]}")
        print(f"    query:  {ex['query'][:240]}")
