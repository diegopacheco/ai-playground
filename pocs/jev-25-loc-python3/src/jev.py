import warnings

import numpy
from llama_cpp import Llama

REPO_ID = "Qwen/Qwen3-0.6B-GGUF"
FILENAME = "Qwen3-0.6B-Q8_0.gguf"
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def load_model():
    return Llama.from_pretrained(
        repo_id=REPO_ID,
        filename=FILENAME,
        n_ctx=512,
        logits_all=True,
        verbose=False,
    )


def build_prompt(text, choices):
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, choices))
    return f"""<|im_start|>system
Choose one option.<|im_end|>
<|im_start|>user
Email: {text}\n\n{options}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n"""


def to_probabilities(choice_logits):
    logprobs = choice_logits - numpy.logaddexp.reduce(choice_logits)
    return logprobs, numpy.exp(logprobs)


def classify(model, text, choices):
    labels = LABELS[: len(choices)]
    model.reset()
    model.eval(tokens=model.tokenize(text=build_prompt(text, choices).encode(), add_bos=False, special=True))
    logits = model.scores[model.n_tokens - 1]
    token_ids = [model.tokenize(text=label.encode(), add_bos=False)[0] for label in labels]
    choice_logits = numpy.asarray([logits[token_id] for token_id in token_ids])
    logprobs, probabilities = to_probabilities(choice_logits)
    return {
        "Logits": choice_logits,
        "Log probabilities": logprobs,
        "Probabilities": probabilities,
    }
