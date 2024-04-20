from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

input_text = "I feel great today because [MASK]."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

token_logits = model(input_ids).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(input_text.replace(tokenizer.mask_token, tokenizer.decode([token])))