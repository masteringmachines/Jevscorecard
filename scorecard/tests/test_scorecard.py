import contextlib
import io
import json
import os
import sys
import tempfile
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scorecard.cache import Cache
from scorecard.cli import main, parse_weights, score_candidates
from scorecard.client import JevError, make_http_transport
from scorecard.rubric import RubricError, build_questions, parse_rubric
from scorecard.scoring import evaluate, rank, sensitivity, total

RUBRIC = parse_rubric({
    "name": "demo",
    "dimensions": [
        {"id": "a", "question": "How good is A?", "levels": ["bad", "ok", "good"], "weight": 1},
        {"id": "b", "question": "How good is B?", "levels": ["bad", "ok", "good"], "weight": 1},
    ],
    "gates": [
        {"id": "has_ask", "question": "Is there an ask?", "threshold": 0.5},
        {"id": "hype", "question": "Is it hype?", "threshold": 0.7, "must_be": False, "required": False},
    ],
})


def response(a=1.0, b=1.0, conf=0.9, ask=0.9, hype=0.1, model="jev-1.13.0"):
    """A response shaped like the API reference's examples."""
    return {
        "model": model,
        "answers": {
            "a": {"type": "score", "score": a, "legend": {"0": "bad", "1": "ok", "2": "good"},
                  "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "confidence": conf},
            "b": {"type": "score", "score": b, "legend": {"0": "bad", "1": "ok", "2": "good"},
                  "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "confidence": conf},
            "has_ask": {"type": "noul", "noul": ask},
            "hype": {"type": "noul", "noul": hype},
        },
        "usage": {"input_tokens": 300, "output_tokens": 20},
    }


# ---------- rubric ----------

def test_rubric_rejects_bad_input():
    bad = [
        {"dimensions": []},
        {"dimensions": [{"id": "x", "question": "q", "levels": ["only one"]}]},
        {"dimensions": [{"id": "x", "question": "q", "levels": ["a", "b"], "weight": 0}]},
        {"dimensions": [{"id": "x", "question": "q", "levels": ["a", "b"]}],
         "gates": [{"id": "x", "question": "dup id"}]},
        {"dimensions": [{"id": "x", "question": "q", "levels": ["a", "b"]}],
         "gates": [{"id": "g", "question": "q", "threshold": 1.5}]},
    ]
    for data in bad:
        try:
            parse_rubric(data)
        except RubricError:
            continue
        raise AssertionError(f"expected RubricError for {data}")


def test_build_questions_matches_api_shape():
    q = build_questions(RUBRIC)
    assert q["a"] == {"type": "score", "instructions": "How good is A?", "criteria": ["bad", "ok", "good"]}
    assert q["has_ask"] == {"type": "noul", "instructions": "Is there an ask?"}


# ---------- scoring ----------

def test_total_is_weighted_mean_scaled_to_100():
    c = evaluate(RUBRIC, "x", response(a=2.0, b=0.0))
    assert abs(total(c, {"a": 1, "b": 1}) - 50.0) < 1e-9
    assert abs(total(c, {"a": 3, "b": 1}) - 75.0) < 1e-9  # a (norm 1.0) now counts 3x


def test_score_between_levels_is_normalized_and_clamped():
    c = evaluate(RUBRIC, "x", response(a=1.5, b=9.0))
    assert abs(c.dims["a"].norm - 0.75) < 1e-9
    assert c.dims["b"].norm == 1.0  # clamped, never above 1


def test_failed_required_gate_disqualifies_and_ranks_last():
    good = evaluate(RUBRIC, "good", response(a=1.0, b=1.0))
    bad = evaluate(RUBRIC, "bad", response(a=2.0, b=2.0, ask=0.1))  # higher scores but no ask
    ranked = rank([bad, good], {"a": 1, "b": 1})
    assert [c.name for c in ranked] == ["good", "bad"]
    assert ranked[1].disqualified


def test_must_be_false_gate_and_non_required_gate():
    # 'hype' must be NO, but it is not required: failing it flags nothing disqualifying.
    c = evaluate(RUBRIC, "x", response(hype=0.95))
    assert not c.gates["hype"].passed
    assert not c.disqualified


def test_low_confidence_and_borderline_are_flagged():
    c = evaluate(RUBRIC, "x", response(conf=0.4, ask=0.55), min_confidence=0.6)
    assert any("low confidence on a" in f for f in c.flags)
    assert any("borderline gate has_ask" in f for f in c.flags)
    clean = evaluate(RUBRIC, "y", response(conf=0.95, ask=0.95, hype=0.05))
    assert clean.flags == []


def test_missing_or_malformed_answer_raises():
    r = response()
    del r["answers"]["a"]
    try:
        evaluate(RUBRIC, "x", r)
    except JevError:
        pass
    else:
        raise AssertionError("expected JevError for a missing answer")


def test_sensitivity_detects_a_flip_and_stability():
    # p is strong on 'a', q is strong on 'b'; equal weights -> tie broken by name.
    p = evaluate(RUBRIC, "p", response(a=2.0, b=0.0))
    q = evaluate(RUBRIC, "q", response(a=0.0, b=2.0))
    flipping = sensitivity([p, q], {"a": 1, "b": 1})
    assert "NOT stable" in flipping[0]
    assert any("'b' weight x2" in line for line in flipping)

    dominant = evaluate(RUBRIC, "dominant", response(a=2.0, b=2.0))
    weak = evaluate(RUBRIC, "weak", response(a=0.0, b=0.0))
    assert "stable" in sensitivity([dominant, weak], {"a": 1, "b": 1})[0]


def test_parse_weights():
    base = {"a": 1.0, "b": 1.0}
    assert parse_weights("a=3", base) == {"a": 3.0, "b": 1.0}
    for bad in ("zzz=1", "a=oops", "a=-2"):
        try:
            parse_weights(bad, base)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# ---------- caching: re-weighting must cost zero calls ----------

def test_cache_hit_skips_transport_and_reweighting_needs_no_calls():
    calls = []

    def transport(payload):
        calls.append(payload)
        return response(a=2.0, b=0.0)

    cache = Cache(None)
    items = [("one.txt", "text one"), ("two.txt", "text two")]
    first, n1 = score_candidates(RUBRIC, items, "jev-1.13.0", transport, cache, 0.6, 2)
    assert n1 == 2 and len(calls) == 2

    again, n2 = score_candidates(RUBRIC, items, "jev-1.13.0", transport, cache, 0.6, 2)
    assert n2 == 0 and len(calls) == 2  # all answered from cache

    # Different weights re-rank from the same evaluated answers, no transport involved.
    assert total(first[0], {"a": 1, "b": 1}) != total(first[0], {"a": 5, "b": 1})

    # Changing the model changes the key, so the model is asked again.
    _, n3 = score_candidates(RUBRIC, items, "jev-1.14.0", transport, cache, 0.6, 2)
    assert n3 == 2


def test_cache_persists_to_disk_and_survives_corruption():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cache.json")
        c = Cache(path)
        c.put("k", {"v": 1})
        c.save()
        assert Cache(path).get("k") == {"v": 1}
        with open(path, "w") as f:
            f.write("{not json")
        assert Cache(path).get("k") is None  # corrupt cache behaves as empty


# ---------- HTTP client: retry behavior, with no real network ----------

class FakeResp:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b'{"error":"x"}'))


def test_client_retries_429_then_succeeds():
    attempts = []
    sleeps = []

    def opener(request, timeout):
        attempts.append(request)
        if len(attempts) < 3:
            raise http_error(429)
        return FakeResp(response())

    t = make_http_transport("key", opener=opener, sleep=sleeps.append)
    out = t({"state": "s", "model": "m", "questions": {}})
    assert out["model"] == "jev-1.13.0"
    assert len(attempts) == 3 and sleeps == [1.0, 2.0]  # exponential backoff
    req = attempts[0]
    assert req.get_header("Authorization") == "Bearer key"
    assert req.full_url.endswith("/v1/systemone")


def test_client_does_not_retry_auth_errors():
    attempts = []

    def opener(request, timeout):
        attempts.append(1)
        raise http_error(401)

    t = make_http_transport("bad", opener=opener, sleep=lambda s: None)
    try:
        t({})
    except JevError as err:
        assert "401" in str(err)
    else:
        raise AssertionError("expected JevError")
    assert len(attempts) == 1


def test_client_gives_up_after_retries():
    t = make_http_transport("k", retries=2, opener=lambda r, timeout: (_ for _ in ()).throw(http_error(529)),
                            sleep=lambda s: None)
    try:
        t({})
    except JevError as err:
        assert "giving up after 3 attempts" in str(err)
    else:
        raise AssertionError("expected JevError")


# ---------- CLI end to end (mock mode: no key, no network) ----------

def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(args))
    return code, out.getvalue(), err.getvalue()


def test_cli_mock_run_labels_itself_and_lists_all_candidates():
    here = os.path.dirname(__file__)
    rubric = os.path.join(here, "..", "examples", "pitch_rubric.json")
    pitches = [os.path.join(here, "..", "examples", "pitches", n) for n in ("acme.txt", "bolt.txt", "cora.txt")]
    code, out, _ = run_cli(rubric, *pitches, "--mock", "--sensitivity")
    assert code == 0
    assert "MOCK MODE" in out
    for name in ("acme.txt", "bolt.txt", "cora.txt"):
        assert name in out


def test_cli_json_output_is_valid_and_weights_apply():
    here = os.path.dirname(__file__)
    rubric = os.path.join(here, "..", "examples", "pitch_rubric.json")
    pitch = os.path.join(here, "..", "examples", "pitches", "acme.txt")
    code, out, _ = run_cli(rubric, pitch, "--mock", "--json", "--weights", "market=9")
    assert code == 0
    data = json.loads(out)
    assert data["weights"]["market"] == 9.0
    assert data["ranking"][0]["name"] == "acme.txt"


def test_cli_dry_run_prints_payload_without_a_key():
    here = os.path.dirname(__file__)
    rubric = os.path.join(here, "..", "examples", "pitch_rubric.json")
    pitch = os.path.join(here, "..", "examples", "pitches", "acme.txt")
    old = os.environ.pop("TYPESAFE_API_KEY", None)
    try:
        code, out, _ = run_cli(rubric, pitch, "--dry-run", "--model", "jev-1.13.0")
    finally:
        if old is not None:
            os.environ["TYPESAFE_API_KEY"] = old
    payload = json.loads(out)
    assert code == 0 and payload["model"] == "jev-1.13.0"
    assert set(payload["questions"]) == {"clarity", "market", "traction", "has_ask", "unsupported_superlatives"}


def test_cli_missing_key_is_a_clear_error():
    here = os.path.dirname(__file__)
    rubric = os.path.join(here, "..", "examples", "pitch_rubric.json")
    pitch = os.path.join(here, "..", "examples", "pitches", "acme.txt")
    old = os.environ.pop("TYPESAFE_API_KEY", None)
    try:
        code, _, err = run_cli(rubric, pitch)
    finally:
        if old is not None:
            os.environ["TYPESAFE_API_KEY"] = old
    assert code == 2 and "TYPESAFE_API_KEY" in err


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"All {len(tests)} tests passed.")
