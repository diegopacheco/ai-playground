# TinyLlama Agent Router

Cheap classification and routing. Thirty labeled customer messages get routed to
one of five agents (`billing`, `tech_support`, `sales`, `abuse`, `general`) and
tagged with a sentiment (`positive`, `neutral`, `negative`), entirely locally.

Routing is the classic case for a tiny model: it runs on every inbound message,
so per-call cost dominates, and the output space is a handful of labels. This
POC measures whether TinyLlama is actually good enough for it.

Both decisions use Ollama structured output with a JSON-schema `enum`, so the
model is constrained to emit a valid label. Parsing can never fail, and the
router cannot invent an agent that does not exist.

Agent and sentiment are two separate single-decision calls. Asking for both in
one call scored worse (agent accuracy 0.37 vs 0.43).

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

Per-request output is on by default. `VERBOSE=0 ./run.sh` prints the summary only.

## Result

```
model tinyllama, 30 labeled requests

------------------------------------------------------------------
agent accuracy      0.43  (13/30)
sentiment accuracy  0.60  (18/30)
both correct        0.30  (9/30)
serial latency      132 ms/request -> 7.6 req/sec

agent confusion (rows expected, columns predicted)
                billin  tech_s   sales   abuse  genera
billing              6       1       0       0       0
tech_support         0       7       0       0       0
sales                2       4       0       0       0
abuse                4       1       0       0       0
general              4       1       0       0       0

8 workers        14.8 req/sec (2.0x over serial)
extrapolated        1.28M requests/day on this one machine, zero API cost
```

## Reading the result

The confusion matrix is the whole story. **Three of the five columns are empty.**
TinyLlama never once answers `sales`, `abuse` or `general` — it routes every
message to `billing` or `tech_support`, the first two entries in the enum. It
gets 6/7 billing and 7/7 tech_support right, and 0/16 of everything else.

That is not a knowledge gap, it is the model failing to discriminate across a
5-way choice at all. It anchors on the first labels it saw and stays there.
Sentiment, a 3-way choice, does better at 0.60 but is still near the level you
would get by always answering `negative`.

Prompt shape moves it a little and does not fix it:

```
agent-only, 0-shot   acc=0.23   predictions: {billing: 30}
agent-only, 5-shot   acc=0.43   predictions: {billing: 16, tech_support: 14}
agent-only, 10-shot  acc=0.33   predictions: {billing: 24, tech_support: 5, sales: 1}
```

**The throughput is real, the accuracy is not.** 1.28M requests/day on one
machine at zero API cost is exactly the economic argument for tiny models — but
a router that silently drops sales leads and spam into the billing queue is
worse than no router.

### Same code, llama3.2

```
model llama3.2, 30 labeled requests

------------------------------------------------------------------
agent accuracy      0.93  (28/30)
sentiment accuracy  0.87  (26/30)
both correct        0.80  (24/30)
serial latency      365 ms/request -> 2.7 req/sec

agent confusion (rows expected, columns predicted)
                billin  tech_s   sales   abuse  genera
billing              7       0       0       0       0
tech_support         1       6       0       0       0
sales                0       0       6       0       0
abuse                1       0       0       4       0
general              0       0       0       0       5

8 workers        6.8 req/sec (2.5x over serial)
extrapolated        0.59M requests/day on this one machine, zero API cost
```

```bash
MODEL=llama3.2 ./run.sh
```

All five columns populated, 0.93 agent accuracy, for 2.8x the latency and half
the daily throughput. Still ~590k requests/day locally at zero API cost, which
keeps the entire economic argument intact.

The conclusion is not "small models cannot route." It is that the floor is
higher than 1.1B. TinyLlama would need a fine-tune on this label set to compete;
llama3.2 does it zero-effort.

## Configuration

| Variable  | Default     | Meaning                             |
|-----------|-------------|-------------------------------------|
| `MODEL`   | `tinyllama` | Ollama model to evaluate            |
| `WORKERS` | `8`         | threads for the concurrency test    |
| `VERBOSE` | `1`         | `0` prints the summary table only   |

## Layout

```
src/requests_data.py  30 labeled requests, 5 agents, 3 sentiments
src/router.py         enum-constrained agent and sentiment calls, few-shot
src/main.py           accuracy, confusion matrix, serial and threaded throughput
```

## Notes

The extrapolated requests/day figure assumes the measured concurrent rate holds
for 24 hours on this machine, with no batching and no GPU sharing. It is a
back-of-envelope ceiling, not a benchmark.
