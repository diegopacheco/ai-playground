import os
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, logging)

logging.set_verbosity_error()

TARGET = os.environ.get("TARGET_MODEL", "NousResearch/Llama-2-7b-chat-hf")
DRAFT = os.environ.get("DRAFT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "96"))
DRAFT_TOKENS = int(os.environ.get("DRAFT_TOKENS", "8"))
CONTEXT_LIMIT = 2048

PASSAGE = (
    "A write-ahead log is an append-only file. Every change is written to the log "
    "before it is applied to the main data files. If the process crashes, recovery "
    "replays the log from the last checkpoint and the database returns to a "
    "consistent state."
)

PROMPTS = [
    ("copy (high acceptance)",
     f"[INST] Repeat the following text exactly, word for word:\n\n{PASSAGE} [/INST] "
     f"Sure, here it is:\n\n"),
    ("extract (high acceptance)",
     f"[INST] Copy the sentence from this passage that mentions crashes, word for "
     f"word:\n\n{PASSAGE} [/INST]"),
    ("open ended (low acceptance)",
     "[INST] Explain what a write-ahead log is and why databases use one. [/INST]"),
    ("code (medium acceptance)",
     "[INST] Write a Python function that merges two sorted lists. [/INST]"),
]


class TokenLimit(StoppingCriteria):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, input_ids, scores, **kwargs):
        done = input_ids.shape[-1] >= self.limit
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool,
                          device=input_ids.device)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(name, dev):
    print(f"loading {name}")
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(dev).eval()


def generate(model, inputs, assistant=None):
    prompt_length = inputs["input_ids"].shape[1]
    kwargs = {}
    if assistant is not None:
        kwargs["assistant_model"] = assistant
        kwargs["num_assistant_tokens"] = DRAFT_TOKENS
        kwargs["num_assistant_tokens_schedule"] = "constant"
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=CONTEXT_LIMIT,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList(
                [TokenLimit(prompt_length + MAX_NEW_TOKENS)]),
            **kwargs,
        )
    elapsed = time.perf_counter() - start
    return output[0][prompt_length:prompt_length + MAX_NEW_TOKENS], elapsed


def main():
    dev = device()
    print(f"device {dev}, {MAX_NEW_TOKENS} new tokens, "
          f"{DRAFT_TOKENS} draft tokens per step\n")

    target_tokenizer = AutoTokenizer.from_pretrained(TARGET)
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT)
    target_vocab = target_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    conflicts = [token for token, token_id in draft_vocab.items()
                 if target_vocab.get(token) != token_id]
    assert not conflicts, \
        f"draft and target disagree on {len(conflicts)} token ids, cannot speculate"
    print(f"tokenizers compatible: {len(draft_vocab)} shared token ids, "
          f"{len(target_vocab) - len(draft_vocab)} extra in target\n")

    target = load(TARGET, dev)
    draft = load(DRAFT, dev)

    warmup = target_tokenizer("warm up", return_tensors="pt").to(dev)
    generate(target, warmup)
    generate(target, warmup, assistant=draft)
    generate(draft, warmup)
    print()

    solo_tokens, solo_seconds = generate(draft, target_tokenizer(
        PROMPTS[2][1], return_tensors="pt").to(dev))
    target_tokens, target_seconds = generate(target, target_tokenizer(
        PROMPTS[2][1], return_tensors="pt").to(dev))
    draft_rate = solo_tokens.shape[0] / solo_seconds
    target_rate = target_tokens.shape[0] / target_seconds
    print(f"draft alone   {draft_rate:6.2f} tok/s")
    print(f"target alone  {target_rate:6.2f} tok/s")
    print(f"speed ratio   {draft_rate / target_rate:6.2f}x "
          f"(speculation needs this to be large)\n")

    baseline_total = assisted_total = 0.0
    token_total = 0
    identical = 0

    for label, prompt in PROMPTS:
        inputs = target_tokenizer(prompt, return_tensors="pt").to(dev)
        baseline_tokens, baseline_seconds = generate(target, inputs)
        assisted_tokens, assisted_seconds = generate(target, inputs, assistant=draft)

        count = min(baseline_tokens.shape[0], assisted_tokens.shape[0])
        same = torch.equal(baseline_tokens[:count], assisted_tokens[:count])
        identical += same
        baseline_total += baseline_seconds
        assisted_total += assisted_seconds
        token_total += count

        print(f"{label}")
        print(f"  target only     {baseline_seconds:6.2f}s  "
              f"{count / baseline_seconds:6.2f} tok/s")
        print(f"  with tinyllama  {assisted_seconds:6.2f}s  "
              f"{count / assisted_seconds:6.2f} tok/s  "
              f"{baseline_seconds / assisted_seconds:5.2f}x")
        print(f"  identical output: {same}")
        print(f"  {target_tokenizer.decode(assisted_tokens, skip_special_tokens=True)[:150].strip()}")
        print()

    print("-" * 70)
    print(f"target only     {token_total / baseline_total:6.2f} tok/s")
    print(f"with tinyllama  {token_total / assisted_total:6.2f} tok/s")
    print(f"overall         {baseline_total / assisted_total:6.2f}x")
    print(f"identical outputs {identical}/{len(PROMPTS)}")


if __name__ == "__main__":
    main()
