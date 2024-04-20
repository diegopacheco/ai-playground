from transformers import RobertaTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

print(tokenizer("Hello world")["input_ids"])
print(tokenizer(" Hello world")["input_ids"])

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(outputs)
print(last_hidden_states.shape)

# print outpus in text format
print(tokenizer.decode(inputs["input_ids"][0]))