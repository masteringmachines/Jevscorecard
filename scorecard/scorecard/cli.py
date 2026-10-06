"""
cli.py — scorecard RUBRIC FILE [FILE ...]

    scorecard rubric.json pitches/*.txt
    scorecard rubric.json pitches/*.txt --weights market=4,clarity=1   # free: uses cached answers
    scorecard rubric.json pitches/*.txt --sensitivity
    scorecard rubric.json pitches/*.txt --mock                         # no API key, no network
    scorecard rubric.json pitches/acme.txt --dry-run                   # print the request, send nothing
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from .cache import Cache
from .client import DEFAULT_MODEL, JevError, Transport, transport_from_env
from .mock import mock_transport
from .report import format_table, to_json
from .rubric import RubricError, build_questions, load_rubric
from .scoring import Candidate, evaluate, rank, sensitivity


def parse_weights(spec: str | None, known: dict[str, float]) -> dict[str, float]:
    weights = dict(known)
    if not spec:
        return weights
    for part in spec.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in weights:
            raise ValueError(f"unknown dimension in --weights: '{key}' (have: {', '.join(weights)})")
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"bad weight for '{key}': '{value}'") from None
        if number <= 0:
            raise ValueError(f"weight for '{key}' must be greater than 0")
        weights[key] = number
    return weights


def read_candidates(paths: list[str]) -> list[tuple[str, str]]:
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            out.append((os.path.basename(path), f.read()))
    return out


def score_candidates(rubric, candidates, model, transport: Transport, cache: Cache,
                     min_confidence: float, workers: int) -> tuple[list[Candidate], int]:
    """Returns (evaluated candidates, number of real model calls made)."""
    questions = build_questions(rubric)

    def one(item: tuple[str, str]) -> tuple[Candidate, bool]:
        name, text = item
        payload = {"state": text, "model": model, "questions": questions}
        key = Cache.key(payload)
        response = cache.get(key)
        called = False
        if response is None:
            response = transport(payload)
            cache.put(key, response)
            called = True
        return evaluate(rubric, name, response, min_confidence), called

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, candidates))
    calls = sum(1 for _, called in results if called)
    return [c for c, _ in results], calls


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scorecard", description="Rank text files against a weighted rubric using Jev.")
    p.add_argument("rubric", help="rubric JSON file")
    p.add_argument("files", nargs="+", help="text files to score")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model id; pin a version (e.g. jev-1.13.0) when comparing runs")
    p.add_argument("--weights", help="override weights, e.g. market=4,clarity=1 (reuses cached answers)")
    p.add_argument("--min-confidence", type=float, default=0.6, help="flag score answers below this confidence (default 0.6)")
    p.add_argument("--sensitivity", action="store_true", help="check whether the top pick survives weight changes")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--cache", default=".scorecard-cache.json", help="cache file path")
    p.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    p.add_argument("--workers", type=int, default=4, help="concurrent requests (default 4)")
    p.add_argument("--mock", action="store_true", help="use a pseudo-random stand-in instead of Jev (offline demo)")
    p.add_argument("--dry-run", action="store_true", help="print the first request payload and exit")
    args = p.parse_args(argv)

    try:
        rubric = load_rubric(args.rubric)
        weights = parse_weights(args.weights, rubric.default_weights())
        candidates = read_candidates(args.files)

        if args.dry_run:
            name, text = candidates[0]
            print(json.dumps({"state": text, "model": args.model, "questions": build_questions(rubric)}, indent=2))
            return 0

        transport = mock_transport if args.mock else transport_from_env()
        cache = Cache(None if (args.no_cache or args.mock) else args.cache)
        evaluated, calls = score_candidates(rubric, candidates, args.model, transport, cache,
                                            args.min_confidence, args.workers)
        cache.save()
    except (RubricError, JevError, ValueError, OSError, json.JSONDecodeError) as err:
        print(f"scorecard: {err}", file=sys.stderr)
        return 2

    ranked = rank(evaluated, weights)
    if args.json:
        print(to_json(rubric, ranked, weights))
    else:
        print(format_table(rubric, ranked, weights, mock=args.mock))
        if args.sensitivity:
            print("\n" + "\n".join(sensitivity(evaluated, weights)))
        print(f"\n{calls} model call(s) made, {len(evaluated) - calls} answered from cache.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
