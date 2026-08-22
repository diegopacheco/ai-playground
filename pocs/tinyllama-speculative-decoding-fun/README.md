# TinyLlama Speculative Decoding

TinyLlama's most serious production use. TinyLlama-1.1B drafts tokens fast,
Llama-2-7B verifies them in a single batched forward pass, and the output is
**identical** to what the 7B would have produced on its own.

This works because TinyLlama deliberately reuses Llama-2's architecture and
tokenizer. Same 32000 token ids, so a draft token means the same thing to both
models and no translation layer is needed.

The POC measures four prompts chosen for different acceptance rates, and asserts
token-for-token equality between assisted and unassisted output on every one.

## Requirements

- Python 3.14.6
- ~15 GB disk for the two models, ~18 GB RAM to hold them in fp16
- Apple Silicon (MPS), CUDA, or CPU — detected automatically

## Install dependencies

```bash
./install-deps.sh
```

## Run

```bash
./run.sh
```

First run downloads `NousResearch/Llama-2-7b-chat-hf` (13 GB, an ungated mirror
of Llama-2-7B-chat) and `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (2.2 GB).

## Result

Apple Silicon, MPS, fp16, 96 new tokens, 8 draft tokens per step:

```
device mps, 96 new tokens, 8 draft tokens per step

tokenizers compatible: 32000 shared token ids, 1 extra in target

draft alone    87.60 tok/s
target alone   26.15 tok/s
speed ratio     3.35x (speculation needs this to be large)

copy (high acceptance)
  target only       2.21s   24.83 tok/s
  with tinyllama    1.01s   54.45 tok/s   2.19x
  identical output: True

extract (high acceptance)
  target only       1.75s   24.05 tok/s
  with tinyllama    1.31s   32.16 tok/s   1.34x
  identical output: True

open ended (low acceptance)
  target only       3.79s   25.31 tok/s
  with tinyllama    3.13s   30.63 tok/s   1.21x
  identical output: True

code (medium acceptance)
  target only       3.76s   25.54 tok/s
  with tinyllama    4.66s   20.59 tok/s   0.81x
  identical output: True

----------------------------------------------------------------------
target only      25.10 tok/s
with tinyllama   28.57 tok/s
overall           1.14x
identical outputs 4/4
```

## Reading the result

**Correctness holds everywhere.** 4/4 prompts produced byte-identical tokens
with and without the draft model. That is the guarantee speculative decoding
makes and it is the reason it is safe to turn on in production: it is a pure
latency optimization, not a quality tradeoff.

**Speed depends entirely on acceptance rate.** Copying a passage the model can
see gets 2.19x. Open-ended prose gets 1.21x. Generating code — where TinyLlama
and Llama-2 diverge on almost every identifier — gets 0.81x, an actual
slowdown, because every rejected draft token is wasted work.

**This hardware is the wrong hardware for it.** The draft is only 3.35x faster
than the target here. On a discrete GPU, 7B fp16 decode is purely
memory-bandwidth bound and a 1.1B draft runs 6-10x faster, which is where the
usual "2-3x throughput" figure comes from. Apple Silicon's unified memory makes
small-model decode dominated by per-kernel launch overhead rather than
bandwidth, so the draft never gets proportionally cheap and the margin
collapses. The 1.14x overall here is a floor imposed by the machine, not by the
technique.

**Batch size 1 is also the worst case.** Real serving stacks (vLLM, TensorRT-LLM)
verify many sequences at once, which amortizes the target forward pass and
widens the gap further.

## Two things that will bite you

**1. Use transformers 4.x, not 5.x.** `requirements.txt` pins `transformers<5`
on purpose. On 5.15.1 assisted generation crashes with:

```
RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1, 64]
```

The assistant's KV cache gets cropped by one token per round while the target
may accept up to `k` candidates, so when acceptance is near-perfect the cache
runs ahead of `input_ids` and the next draft prefill gets zero new tokens. It
fires precisely on the high-acceptance prompts that speculative decoding exists
to speed up. 4.57.6 handles this correctly.

**2. Cap length with a stopping criterion, not `max_new_tokens`.**
`_calculate_new_tokens` guards `max_new_tokens == 0` but speculative decoding
can overshoot the cap and drive it *negative*, which slips past the guard and
produces the same empty-prefill crash. `src/main.py` passes a generous
`max_length` plus a `TokenLimit` stopping criterion instead, then trims both
outputs to the same length before comparing them.

## Configuration

| Variable         | Default                             | Meaning                       |
|------------------|-------------------------------------|-------------------------------|
| `TARGET_MODEL`   | `NousResearch/Llama-2-7b-chat-hf`   | the model being verified      |
| `DRAFT_MODEL`    | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`| the model drafting tokens     |
| `MAX_NEW_TOKENS` | `96`                                | tokens generated per prompt   |
| `DRAFT_TOKENS`   | `8`                                 | candidates drafted per step   |

`DRAFT_TOKENS` is the main knob. Higher pays off when acceptance is high and
costs more when it is low:

```
copy prompt, 64 tokens:   k=2 -> 12.1 tok/s   k=3 -> 16.6   k=4 -> 45.1   k=8 -> 57.7
```

## Layout

```
src/main.py   tokenizer compatibility check, per-prompt benchmark, equality assertion
```

## Notes

The tokenizer check is a subset test, not equality: the NousResearch mirror adds
a `<pad>` token at id 32000 that upstream TinyLlama does not have. What matters
is that no token id means different things to the two models, and zero ids
conflict.

`num_assistant_tokens_schedule="constant"` is set explicitly. The default
heuristic schedule adapts `k` at runtime and can drive it to zero, which is one
more way into the empty-prefill crash above.
