import torch
import matplotlib.pyplot as plt
from transformers import XLMWithLMHeadModel, XLMTokenizer

# Load pre-trained model and tokenizer
model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')

# Define a function to translate text
def translate_text(text, src_lang, tgt_lang):
    input_ids = tokenizer.encode(src_lang + '-' + tgt_lang + ' ' + text, return_tensors='pt')
    attention_mask = tokenizer.encode(src_lang + '-' + tgt_lang + ' ' + text, return_tensors='pt', max_length=50, padding='max_length', truncation=True)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Translate a sample text from English to Spanish
src_lang = 'en'
tgt_lang = 'es'
text = 'Hello, how are you?'
translated_text = translate_text(text, src_lang, tgt_lang)
print(f'Translated text: {translated_text}')

def plot_results(src_text, tgt_text):
    # Convert tensors to lists
    src_text = src_text[0].tolist()
    tgt_text = tgt_text[0].tolist()

    # Decode the tokens back into words
    src_text = [tokenizer.decode([token]) for token in src_text]
    tgt_text = [tokenizer.decode([token]) for token in tgt_text]

    # Count the occurrences of each word
    src_counts = {word: src_text.count(word) for word in set(src_text)}
    tgt_counts = {word: tgt_text.count(word) for word in set(tgt_text)}

    # Create the plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot source text
    ax[0].bar(src_counts.keys(), src_counts.values())
    ax[0].set_title('Source Text')
    ax[0].set_xlabel('Word')
    ax[0].set_ylabel('Count')

    # Plot target text
    ax[1].bar(tgt_counts.keys(), tgt_counts.values())
    ax[1].set_title('Target Text')
    ax[1].set_xlabel('Word')
    ax[1].set_ylabel('Count')

    # Show the plot
    plt.tight_layout()
    plt.show()

src_text = tokenizer.encode(text, return_tensors='pt')
tgt_text = tokenizer.encode(translated_text, return_tensors='pt')
plot_results(src_text, tgt_text)