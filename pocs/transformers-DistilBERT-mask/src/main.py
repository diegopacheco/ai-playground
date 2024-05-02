from transformers import AutoTokenizer, DistilBertForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

text = "The capital of France, which is known for the Eiffel Tower, is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
print(f"input text: {text}")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

# decode the predicted token id
predicted_token = tokenizer.decode(predicted_token_id.item())
print(f"predicted token: {predicted_token}")