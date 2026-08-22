# TinyLlama PII Detection

Cheap PII classification with a 1.1B model. Forty labeled texts, six PII labels
(`EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`, `IP_ADDRESS`, `PERSON_NAME`), and three
detectors scored against the same ground truth:

1. **regex** — deterministic patterns, no model at all
2. **llm all labels** — TinyLlama asked for all six labels in one constrained call
3. **regex + llm name** — regex for the patterns, TinyLlama only for `PERSON_NAME`

`PERSON_NAME` is the label regex cannot do, so it is the only place a model is
worth paying for.

Both LLM detectors use Ollama's structured output, so the model can only emit
booleans for the allowed labels. It cannot invent a label or return broken JSON.

## Requirements

- Python 3.14.6
- [Ollama](https://ollama.com) running locally

```bash
ollama pull tinyllama
ollama pull llama3.2   # optional, for the comparison below
```

## Install dependencies

```bash
./install-deps.sh
```

## Run

```bash
./run.sh
```

Per-text output is on by default. `VERBOSE=0 ./run.sh` prints the summary only.

## Result

```
model tinyllama, 40 labeled texts

------------------------------------------------------------------------------
regex              precision 1.00  recall 0.72  f1 0.84  exact 30/40     0.0 ms/text
llm all labels     precision 0.27  recall 0.50  f1 0.35  exact 15/40   214.0 ms/text
regex + llm name   precision 0.70  recall 0.97  f1 0.81  exact 24/40    61.4 ms/text

best f1: regex
throughput at regex: 36267.3 texts/sec, single process
```

## Reading the result

- **Regex wins outright on this set.** Precision 1.00, and the only thing it
  misses is `PERSON_NAME`, which drags recall to 0.72.
- **TinyLlama on all six labels is unusable.** f1 0.35. It says yes to nearly
  everything; asking a 1.1B model to reason about six independent labels at once
  is past what it can do.
- **The hybrid is the interesting one.** Handing TinyLlama the single yes/no
  question it can almost answer lifts recall from 0.72 to 0.97 — it catches
  every name — but precision falls to 0.70 because it also flags names in texts
  that have none. Net f1 0.81, slightly below plain regex.

The honest conclusion for TinyLlama: narrow the question until it is binary, and
even then it is a recall booster with a false-positive tax, not a detector. That
shape is fine if a human or a stricter second pass reviews the hits.

### Prompt sensitivity

The name classifier is 4-shot. The number of examples matters more than it
should:

```
0-shot acc=0.25    2-shot acc=0.25    4-shot acc=0.60    6-shot acc=0.25    8-shot acc=0.33
```

With 0, 2 or 6 shots the model answers `true` for all 40 texts. Only the 4-shot
arrangement makes it discriminate at all. This fragility is the model, not the
prompt engineering.

### Same code, llama3.2

```
model llama3.2, 40 labeled texts

------------------------------------------------------------------------------
regex              precision 1.00  recall 0.72  f1 0.84  exact 30/40     0.0 ms/text
llm all labels     precision 0.92  recall 1.00  f1 0.96  exact 37/40   435.4 ms/text
regex + llm name   precision 0.97  recall 0.97  f1 0.97  exact 38/40   171.9 ms/text
```

```bash
MODEL=llama3.2 ./run.sh
```

A 3B model handles all six labels in one call at f1 0.96 and the hybrid reaches
0.97, at roughly 2-3x the latency. If the task is PII detection and not a
TinyLlama study, use the bigger small model.

## Configuration

| Variable  | Default     | Meaning                          |
|-----------|-------------|----------------------------------|
| `MODEL`   | `tinyllama` | Ollama model to evaluate         |
| `VERBOSE` | `1`         | `0` prints the summary table only |

## Layout

```
src/samples.py    40 labeled texts, 12 of them clean
src/detectors.py  regex patterns, constrained-JSON detectors, hybrid
src/main.py       scoring (micro precision/recall/f1, exact match, latency)
```

## Notes

Metrics are micro-averaged over labels. `exact` counts texts where the predicted
label set matched the expected set exactly, which is the number that actually
matters for a redaction pipeline: one wrong label means one wrong redaction.
