# TinyLlama Synthetic Data

Synthetic data generation at volume. A plain Python class mints structured
records (names, emails, sex, birth dates, phones, cities, jobs, salaries) and
TinyLlama writes the free-text fields on top of them.

The point of the POC is the gap between the two: structured fields are
essentially free, LLM-written text is not, and TinyLlama's per-sample quality
is mediocre. You use it when you need volume, not prose.

## Requirements

- Python 3.14.6
- [Ollama](https://ollama.com) running locally with `tinyllama` pulled

```bash
ollama pull tinyllama
```

## Install dependencies

```bash
./install-deps.sh
```

Creates a `.venv` with Python 3.14.6. The only dependency is the `ollama`
client; the generator itself is pure standard library.

## Run

```bash
./run.sh
```

## Result

```
checks     : deterministic by seed, emails unique
structured : 200000 rows in 0.99s -> 201,159 rows/sec
structured : people.jsonl is 56.6 MB
llm        : 5 bios in 1.22s -> 4.11 rows/sec
  Marcelo Gomes: "The designer with a sharp mind and a sophisticated eye, constantly pushing boundaries and elevating the status of his field."
  Olivia Ferreira: "Olivia is a resourceful and experienced Support Agent, always ready to lend a hand and offer assistance to her colleagues in times of need."
  Olivia Teixeira: "Olivia, a dynamic and innovative Software Engineer, always pushing boundaries and bringing about positive change wherever she goes."
  Olivia Imperatriz: A force to be reckoned with.
  Paulo Henriques: "Savoring the bustling streets of Austin, Texas, a stoic and empathetic Sales Rep known only as 'Paulo' navigates the city's social and cultural landscape with
ratio      : free text costs 48,944x more time per row
```

## Reading the result

- 200k rows in about a second, ~57 MB of JSONL. Enough to load a database,
  fuzz an API, or fill a test warehouse.
- TinyLlama produces roughly 4 rows/sec on the same machine. Five orders of
  magnitude slower per row.
- Quality is visibly uneven. Some bios drop the age, some drop the city, one is
  five words long. Fine as filler text, not fine as ground truth.

So: generate structure in Python, and only pay the model for the fields that
genuinely need language.

## Configuration

Environment variables:

| Variable    | Default        | Meaning                          |
|-------------|----------------|----------------------------------|
| `COUNT`     | `200000`       | structured rows to generate      |
| `LLM_COUNT` | `5`            | bios to generate with TinyLlama  |
| `MODEL`     | `tinyllama`    | Ollama model for the bios        |
| `OUT_FILE`  | `people.jsonl` | output path                      |

```bash
COUNT=1000000 LLM_COUNT=0 ./run.sh
```

## Layout

```
src/person.py   PersonGenerator, standard library only, seeded and deterministic
src/main.py     throughput benchmark, sanity checks, TinyLlama enrichment
```

## Notes

`PersonGenerator(seed=N)` is deterministic: the same seed replays the same
dataset, which is what makes generated fixtures usable in tests. Emails carry
the row id so they stay unique across any volume.
