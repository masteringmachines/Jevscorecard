"""
mock.py — a stand-in for Jev so you can try the CLI with no API key.

NOT a model. It derives stable pseudo-random numbers from a hash of the
text and question id, in the same response shape the real API returns.
The rankings it produces mean nothing; it exists so you can see the
mechanics (tables, flags, caching, re-weighting, sensitivity) offline,
and so tests and CI can run the whole pipeline without a network.
"""

from __future__ import annotations
import hashlib
import json


def _unit(*parts: str) -> float:
    """A stable number in [0, 1) derived from the given strings."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def mock_transport(payload: dict) -> dict:
    state = payload["state"] if isinstance(payload["state"], str) else json.dumps(payload["state"])
    answers: dict[str, dict] = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "score":
            levels = len(q["criteria"])
            score = _unit(state, qid, "score") * (levels - 1)
            confidence = 0.35 + 0.6 * _unit(state, qid, "conf")
            lo = int(score)
            hi = min(levels - 1, lo + 1)
            frac = score - lo
            probs = {str(i): 0.0 for i in range(levels)}
            probs[str(lo)] += 1 - frac
            probs[str(hi)] += frac
            answers[qid] = {
                "type": "score",
                "score": round(score, 3),
                "legend": {str(i): str(c) for i, c in enumerate(q["criteria"])},
                "probabilities": probs,
                "confidence": round(confidence, 3),
            }
        elif q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": round(0.5 + 0.5 * _unit(state, qid, "noul"), 3)}
    return {"model": "mock (not Jev)", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}
