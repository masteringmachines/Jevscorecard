"""
rubric.py — the rubric file: what to ask, and how much each answer matters.

A rubric has two kinds of entries:

  dimensions — each becomes a Jev *Score* question (ordered levels you
               write). The answer lands somewhere along those levels and
               is combined into a weighted total.
  gates      — each becomes a Jev *Noul* question (a yes/no probability).
               A gate that fails disqualifies the candidate; a gate that
               is close to its threshold is flagged for a human.

Weights and thresholds live HERE, in plain data you control — never in
a prompt. That's what makes re-weighting free (see scoring.py).
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field


class RubricError(ValueError):
    pass


@dataclass
class Dimension:
    id: str
    question: str
    levels: list[str]
    weight: float = 1.0


@dataclass
class Gate:
    id: str
    question: str
    threshold: float = 0.5   # "yes" means probability >= threshold
    must_be: bool = True     # True: must be yes to pass. False: must be no.
    required: bool = True    # False: informational flag only, never disqualifies


@dataclass
class Rubric:
    name: str
    dimensions: list[Dimension] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)

    def default_weights(self) -> dict[str, float]:
        return {d.id: d.weight for d in self.dimensions}


def parse_rubric(data: dict) -> Rubric:
    if not isinstance(data, dict):
        raise RubricError("rubric must be a JSON object")

    dims: list[Dimension] = []
    for i, raw in enumerate(data.get("dimensions", [])):
        where = f"dimensions[{i}]"
        for key in ("id", "question", "levels"):
            if key not in raw:
                raise RubricError(f"{where} is missing '{key}'")
        levels = raw["levels"]
        # The API accepts 2 to 10 Score levels.
        if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
            raise RubricError(f"{where}.levels must be a list of 2 to 10 descriptions")
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise RubricError(f"{where}.weight must be greater than 0")
        dims.append(Dimension(raw["id"], raw["question"], [str(x) for x in levels], weight))

    gates: list[Gate] = []
    for i, raw in enumerate(data.get("gates", [])):
        where = f"gates[{i}]"
        for key in ("id", "question"):
            if key not in raw:
                raise RubricError(f"{where} is missing '{key}'")
        threshold = float(raw.get("threshold", 0.5))
        if not 0.0 < threshold < 1.0:
            raise RubricError(f"{where}.threshold must be between 0 and 1 (exclusive)")
        gates.append(Gate(
            raw["id"], raw["question"], threshold,
            bool(raw.get("must_be", True)), bool(raw.get("required", True)),
        ))

    if not dims:
        raise RubricError("rubric needs at least one dimension")

    ids = [d.id for d in dims] + [g.id for g in gates]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise RubricError(f"duplicate ids: {', '.join(sorted(dupes))}")

    return Rubric(name=str(data.get("name", "Untitled rubric")), dimensions=dims, gates=gates)


def load_rubric(path: str) -> Rubric:
    with open(path, encoding="utf-8") as f:
        return parse_rubric(json.load(f))


def build_questions(rubric: Rubric) -> dict:
    """The `questions` map for one request. Every dimension and gate is
    asked together: Jev evaluates all questions in a request in parallel
    over the same state, so one call per candidate is enough."""
    questions: dict[str, dict] = {}
    for d in rubric.dimensions:
        questions[d.id] = {"type": "score", "instructions": d.question, "criteria": d.levels}
    for g in rubric.gates:
        questions[g.id] = {"type": "noul", "instructions": g.question}
    return questions
