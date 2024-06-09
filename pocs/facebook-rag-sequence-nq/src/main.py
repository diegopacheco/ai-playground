from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-base")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-base")
model = RagTokenForGeneration.from_pretrained("facebook/rag-sequence-base", retriever=retriever)

input_dict = tokenizer.prepare_seq2seq_batch("who holds the record in 100m freestyle", "michael phelps", return_tensors="pt") 
outputs = model(input_dict["input_ids"], labels=input_dict["labels"])
loss = outputs.loss

print(f"output is {outputs}")
print(f"Loss is {loss.item()}")