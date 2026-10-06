# scorecard

Rank text files against a weighted rubric using **Jev**, TypeSafe AI's
System One model. Zero dependencies, about 600 lines of stdlib Python.

The idea: Jev answers narrow typed questions with probabilities, and
**your code** owns the weights and thresholds. So after one round of
model calls you can re-weight, re-threshold, and stress-test the
ranking instantly, with no further API calls.

```bash
scorecard rubric.json pitches/*.txt                       # one Jev call per file
scorecard rubric.json pitches/*.txt --weights market=5    # free: answers come from cache
scorecard rubric.json pitches/*.txt --sensitivity         # would the #1 pick survive different weights?
```

## What is Jev, here?

Jev is not a text generator. You send a *state* (here, the text of one
candidate) plus typed questions, and get back constrained answers with
probabilities. This tool uses two of the three question types:

| Rubric entry | Jev question | Answer used |
|---|---|---|
| `dimensions` (e.g. "market evidence") | **Score** over levels you write | position along the levels, plus a confidence |
| `gates` (e.g. "states a concrete ask?") | **Noul** (yes/no) | probability of yes |

All questions for a candidate go in **one request**, since Jev evaluates
the questions in a request in parallel over the same state.

## What the code decides

Everything after the model call is ordinary, testable code (`scoring.py`):

- **Total** = weighted mean of the normalized dimension scores, 0 to 100.
- **Gates** disqualify a candidate if a required gate fails (`must_be`
  lets a gate demand "no" instead of "yes"; `required: false` makes it a
  flag only).
- **Needs a human look** if a Score answer's confidence is below
  `--min-confidence` (default 0.6) or a gate probability is within 0.15
  of its threshold.
- **Sensitivity**: halves and doubles each weight in turn and reports
  whether the top pick changes.

## Rubric format

```json
{
  "name": "Seed-stage startup pitch",
  "dimensions": [
    {"id": "market", "weight": 3,
     "question": "How convincing is the evidence that the market is large and the problem is painful?",
     "levels": ["No market evidence", "Assertions only",
                "Some concrete numbers or customer quotes",
                "Strong, specific evidence of a large, painful problem"]}
  ],
  "gates": [
    {"id": "has_ask", "question": "Does the pitch state a specific funding amount or a concrete ask?",
     "threshold": 0.5, "must_be": true}
  ]
}
```

A full example is in `examples/`. Write each question as one quick
judgment about the text, and write level descriptions a person could
apply consistently. Rubrics need 2 to 10 levels per dimension, matching
the API's Score limits.

## Try it without an API key

```bash
python -m scorecard examples/pitch_rubric.json examples/pitches/*.txt --mock --sensitivity
```

`--mock` swaps in a **pseudo-random stand-in**, not a model. Its output
is labeled as mock and its rankings mean nothing. It exists so you can
see the tables, flags, caching, and sensitivity check working offline.
`--dry-run` prints the exact request that would be sent, without sending it.

## Use it for real

```bash
export TYPESAFE_API_KEY=...
scorecard examples/pitch_rubric.json examples/pitches/*.txt --model jev-1.13.0
```

- Honors `TYPESAFE_API_KEY` and `TYPESAFE_BASE_URL`, like the official SDKs.
- **Pin `--model`** to a version when comparing runs; `jev-latest` can move.
- Answers are cached in `.scorecard-cache.json`, keyed by a hash of the
  full request. Weights and thresholds are not part of the request, so
  changing them reuses the cache. Changing the text, a question, a level
  description, or the model asks again.
- 429 and 529 responses are retried with exponential backoff.

## Caveats

- **Not yet run against the live API.** It was written from TypeSafe's
  HTTP API reference and tested with fixed responses shaped like the
  documented examples, plus the mock. Treat your first real run as a
  check, and please open an issue if a response shape differs.
- A well-formed answer is not necessarily a correct one. Validate the
  rubric on a handful of examples you can judge yourself before trusting
  a ranking, and keep a person in the loop for decisions that matter.
  The "needs a human look" flags are a starting point, not a guarantee.
- Jev's hosted model is text-only, and TypeSafe publishes known weak
  spots (such as arithmetic and dates), so prefer questions about
  meaning over questions about counting.
- Not affiliated with TypeSafe AI.

## Layout

```
scorecard/
  rubric.py    load + validate the rubric, build the questions
  client.py    stdlib HTTP client (the only network code), with retry
  cache.py     answer cache that makes re-weighting free
  scoring.py   normalize, weight, gate, rank, sensitivity
  report.py    table and JSON output
  mock.py      offline pseudo-random stand-in
  cli.py       command line
tests/         19 tests, no network or key needed
```

```bash
python tests/test_scorecard.py
```

## License

MIT, see [LICENSE](LICENSE).
