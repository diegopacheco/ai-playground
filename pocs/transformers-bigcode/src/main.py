import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cpu"

model = AutoModelForCausalLM.from_pretrained("bigcode/gpt_bigcode-santacoder", torch_dtype=torch.float32, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("bigcode/gpt_bigcode-santacoder")

def generate(prompt):
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=65, do_sample=False)
    generated_code = tokenizer.batch_decode(generated_ids)[0]
    return generated_code

generated_code = \
            generate("def generate_10_numbers_sort_print():") \
            .replace("<|endoftext|>", "") + \
            "generate_10_numbers_sort_print()"

print(">>> LLM generated code: ")
print(generated_code)

print(">>> LLM generated code result: ")
exec(generated_code)