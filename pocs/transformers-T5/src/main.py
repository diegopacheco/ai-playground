from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids

# print the translation result text
print(tokenizer.decode(model.generate(input_ids).numpy()[0], skip_special_tokens=True))

labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
loss = model(input_ids=input_ids, labels=labels).loss
print(loss.item())

