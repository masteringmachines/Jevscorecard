"""
report.py — render ranked candidates as a table or as JSON.
"""

from __future__ import annotations
import json

from .rubric import Rubric
from .scoring import Candidate, total


def format_table(rubric: Rubric, ranked: list[Candidate], weights: dict[str, float], mock: bool = False) -> str:
    dim_ids = [d.id for d in rubric.dimensions]
    name_w = max([len(c.name) for c in ranked] + [9])
    col_w = max([len(i) for i in dim_ids] + [5])

    models = sorted({c.model for c in ranked})
    lines = []
    if mock:
        lines.append("*** MOCK MODE: pseudo-random numbers, not real Jev answers ***")
    lines.append(f"Rubric: {rubric.name}   model: {', '.join(models)}   candidates: {len(ranked)}")
    lines.append("Weights: " + ", ".join(f"{k}={v:g}" for k, v in weights.items()))
    lines.append("")

    header = f"{'#':>2}  {'candidate':<{name_w}}  {'total':>6}  " + "  ".join(f"{i:>{col_w}}" for i in dim_ids)
    lines.append(header)
    lines.append("-" * len(header))

    place = 0
    for c in ranked:
        if c.disqualified:
            reasons = ", ".join(f"{g.id} (p={g.p_yes:.2f})" for g in c.failed_gates)
            lines.append(f"{'-':>2}  {c.name:<{name_w}}  DISQUALIFIED: failed gate {reasons}")
            continue
        place += 1
        cells = "  ".join(f"{100 * c.dims[i].norm:>{col_w}.0f}" for i in dim_ids)
        lines.append(f"{place:>2}  {c.name:<{name_w}}  {total(c, weights):>6.1f}  {cells}")
        for flag in c.flags:
            lines.append(f"{'':>2}  {'':<{name_w}}  needs a human look: {flag}")
    return "\n".join(lines)


def to_json(rubric: Rubric, ranked: list[Candidate], weights: dict[str, float]) -> str:
    out = []
    for c in ranked:
        out.append({
            "name": c.name,
            "model": c.model,
            "total": None if c.disqualified else round(total(c, weights), 2),
            "disqualified": c.disqualified,
            "dimensions": {
                i: {"score": r.score, "normalized": round(r.norm, 3), "confidence": r.confidence}
                for i, r in c.dims.items()
            },
            "gates": {
                i: {"p_yes": g.p_yes, "passed": g.passed, "borderline": g.borderline}
                for i, g in c.gates.items()
            },
            "flags": c.flags,
        })
    return json.dumps({"rubric": rubric.name, "weights": weights, "ranking": out}, indent=2)
