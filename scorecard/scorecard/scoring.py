"""
scoring.py — turn Jev's answers into a ranking. All plain code.

The model's job ends at "here are typed answers with probabilities".
Everything below — normalizing, weighting, gating, ranking, flagging
what needs a human — is ordinary deterministic code, so you can read
it, test it, and change it without touching a prompt.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .client import JevError
from .rubric import Rubric


@dataclass
class DimResult:
    id: str
    score: float        # position along the levels, 0 .. len(levels)-1
    norm: float         # same position scaled to 0 .. 1
    confidence: float   # from the answer's probability distribution
    review: bool        # True if confidence is below the review cutoff


@dataclass
class GateResult:
    id: str
    p_yes: float
    passed: bool
    borderline: bool
    required: bool


@dataclass
class Candidate:
    name: str
    model: str
    dims: dict[str, DimResult]
    gates: dict[str, GateResult]
    flags: list[str] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[GateResult]:
        return [g for g in self.gates.values() if g.required and not g.passed]

    @property
    def disqualified(self) -> bool:
        return bool(self.failed_gates)


def _answer(answers: dict, qid: str, expected: str) -> dict:
    ans = answers.get(qid)
    if not isinstance(ans, dict) or ans.get("type") != expected:
        raise JevError(f"response is missing a '{expected}' answer for question '{qid}'")
    return ans


def evaluate(rubric: Rubric, name: str, response: dict, min_confidence: float = 0.6) -> Candidate:
    """Parse one response into a Candidate. Raises JevError if an
    expected answer is absent or malformed — better a loud failure than
    a silently-wrong ranking."""
    answers = response.get("answers", {})
    dims: dict[str, DimResult] = {}
    gates: dict[str, GateResult] = {}
    flags: list[str] = []

    for d in rubric.dimensions:
        ans = _answer(answers, d.id, "score")
        try:
            score = float(ans["score"])
            confidence = float(ans["confidence"])
        except (KeyError, TypeError, ValueError) as err:
            raise JevError(f"malformed score answer for '{d.id}'") from err
        top = len(d.levels) - 1
        norm = min(1.0, max(0.0, score / top))
        review = confidence < min_confidence
        dims[d.id] = DimResult(d.id, score, norm, confidence, review)
        if review:
            flags.append(f"low confidence on {d.id} ({confidence:.2f})")

    for g in rubric.gates:
        ans = _answer(answers, g.id, "noul")
        try:
            p = float(ans["noul"])
        except (KeyError, TypeError, ValueError) as err:
            raise JevError(f"malformed noul answer for '{g.id}'") from err
        passed = (p >= g.threshold) == g.must_be
        borderline = abs(p - g.threshold) <= 0.15
        gates[g.id] = GateResult(g.id, p, passed, borderline, g.required)
        if borderline:
            flags.append(f"borderline gate {g.id} (p={p:.2f})")

    return Candidate(name=name, model=str(response.get("model", "?")), dims=dims, gates=gates, flags=flags)


def total(candidate: Candidate, weights: dict[str, float]) -> float:
    """Weighted mean of the normalized dimension scores, as 0..100."""
    weight_sum = sum(weights[d] for d in candidate.dims)
    weighted = sum(weights[d] * r.norm for d, r in candidate.dims.items())
    return 100.0 * weighted / weight_sum


def rank(candidates: list[Candidate], weights: dict[str, float]) -> list[Candidate]:
    """Qualified candidates by total (highest first); disqualified last."""
    ok = [c for c in candidates if not c.disqualified]
    out = [c for c in candidates if c.disqualified]
    ok.sort(key=lambda c: (-total(c, weights), c.name))
    out.sort(key=lambda c: c.name)
    return ok + out


def sensitivity(candidates: list[Candidate], weights: dict[str, float]) -> list[str]:
    """Would the top pick change if any one weight were halved or doubled?

    Free to compute — it only re-runs the arithmetic above on answers
    already in hand, no further model calls."""
    ranked = [c for c in rank(candidates, weights) if not c.disqualified]
    if len(ranked) < 2:
        return ["Sensitivity check needs at least two qualified candidates."]
    base_top = ranked[0].name
    changes: list[str] = []
    for dim_id in weights:
        for factor in (0.5, 2.0):
            trial = dict(weights)
            trial[dim_id] *= factor
            new_top = [c for c in rank(candidates, trial) if not c.disqualified][0].name
            if new_top != base_top:
                changes.append(f"top pick becomes {new_top} if '{dim_id}' weight x{factor:g}")
    if not changes:
        return [f"Top pick '{base_top}' is stable under every single-weight halving/doubling."]
    return [f"Top pick '{base_top}' is NOT stable:"] + [f"  - {c}" for c in changes]
