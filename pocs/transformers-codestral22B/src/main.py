from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('/mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/data-ai/mistral/', 'Codestral-22B-v0.1')
mistral_models_path.mkdir(parents=True, exist_ok=True)
print(f"Will download the model to {mistral_models_path}")
snapshot_download(repo_id="mistralai/Codestral-22B-v0.1", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import FIMRequest

tokenizer = MistralTokenizer.v3()
model = Transformer.from_folder(mistral_models_path)

prompt = "Write a program that generate 100 prime numbers in rust"
request = FIMRequest(prompt=prompt)

tokens = tokenizer.encode_fim(request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=256, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.decode(out_tokens[0])

middle = result.split(prompt)[0].strip()
print(middle)