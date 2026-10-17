"""Rubric check: how often do generated rubrics for memory-irrelevant queries include a
personalization criterion? Decided whether to train on natural rubrics or inject the criterion.
"""

import json
from statistics import mean

import config
from checks.common import get_rubrics, load
from datagen.rubrics import label_personalization
from utils.cache import cached


def run(data_path, regenerate):
    examples, out_dir = load(data_path)
    rubrics = get_rubrics(examples, out_dir, regenerate)
    # Training-set examples carry their labels; others are labeled once and cached.
    labels = {ex["id"]: [c["personalization"] for c in ex["rubric"]] for ex in examples if "rubric" in ex}
    unlabeled = [ex for ex in examples if ex["id"] not in labels]
    if unlabeled:
        labels |= cached(out_dir / "labels.json", unlabeled, lambda ex: label_personalization(rubrics[ex["id"]], ex["memories"], ex["query"]), regenerate)

    rows = []
    for ex in examples:
        rubric, lab = rubrics[ex["id"]], labels[ex["id"]]
        total = sum(c["weight"] for c in rubric)
        rows.append({
            "id": ex["id"],
            "adjacent": ex.get("adjacent", False),
            "n_criteria": len(rubric),
            "n_personalization": sum(lab),
            "personalization_weight_share": sum(c["weight"] for c, l in zip(rubric, lab) if l) / total,
            "personalization_criteria": [c for c, l in zip(rubric, lab) if l],
        })

    def summarize(rs):
        return {
            "n": len(rs),
            "share_with_any": mean(r["n_personalization"] > 0 for r in rs),
            "mean_count": mean(r["n_personalization"] for r in rs),
            "mean_weight_share": mean(r["personalization_weight_share"] for r in rs),
        }

    summary = {"all": summarize(rows)}
    for name, flag in (("adjacent", True), ("not_adjacent", False)):
        rs = [r for r in rows if r["adjacent"] == flag]
        if rs and len(rs) < len(rows):  # only when the data has both kinds
            summary[name] = summarize(rs)

    out = {"data": str(data_path), "rubric_model": config.RUBRIC_MODEL, "summary": summary, "results": rows}
    path = out_dir / "rubric_results.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"\nRubric check — rubric writer and labeler={config.RUBRIC_MODEL}, n={len(rows)}\n")
    print(f"{'subset':<16}{'n':>4}{'any':>8}{'count':>8}{'weight share':>14}")
    for name, s in summary.items():
        print(f"{name:<16}{s['n']:>4}{s['share_with_any']:>8.0%}{s['mean_count']:>8.2f}{s['mean_weight_share']:>14.0%}")
    print("\n(any = share of rubrics with at least one personalization criterion; "
          "weight share = their share of total rubric weight)")
    print(f"\nLabels are LLM-made: hand-check a sample in {out_dir / 'labels.json'} against {out_dir / 'rubrics.json'}.")
    print(f"Full results: {path}")
