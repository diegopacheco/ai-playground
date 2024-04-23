from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
print(tokenizer("Hello world")["input_ids"])
print(tokenizer(" Hello world")["input_ids"])

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
prompt = "What is the main food in Italy? Explain. Dont repeat yourself."
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
gen_tokens = model.generate(input_ids, max_length=50, num_return_sequences=1)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

