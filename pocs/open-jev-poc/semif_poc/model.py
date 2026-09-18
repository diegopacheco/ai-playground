import torch
import transformers

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def download():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL, revision=REVISION)


def load(device: str | None = None):
    device = device or pick_device()
    config = transformers.AutoConfig.from_pretrained(MODEL, revision=REVISION)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = transformers.Qwen3_5ForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        config=config.get_text_config(),
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    metadata = {
        "source": MODEL,
        "revision": REVISION,
        "dtype": "bfloat16",
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata


if __name__ == "__main__":
    download()
