import json
from concurrent.futures import ThreadPoolExecutor


def cached(path, examples, make, regenerate=False, workers=16):
    """Build {example id: make(example)} once and reuse it, so reruns see identical inputs."""
    cache = {} if regenerate or not path.exists() else json.loads(path.read_text())
    missing = [ex for ex in examples if ex["id"] not in cache]
    with ThreadPoolExecutor(workers) as pool:
        for ex, value in zip(missing, pool.map(make, missing)):
            cache[ex["id"]] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))
    return cache
