from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the tokenizer and the model
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load the imdb dataset - only get the first 100 examples
dataset = load_dataset('imdb')['train'].select(range(100))

# Tokenize the dataset
def tokenize_function(examples):
    # Encode the text and ensure that the output length is truncated/padded to a maximum length of 512 tokens
    output = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    
    # The labels should be the same as the input ids, as the model needs to predict the next token in the sequence
    output['labels'] = output['input_ids'].copy()
    
    return output

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=500,
    evaluation_strategy="no",
)

# Create the trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Print the training loss
print(f"Training loss: {trainer.state.best_metric}")

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')

# Extract the training loss values
training_loss = [log['loss'] for log in trainer.state.log_history]

# Plot the training loss
plt.plot(training_loss)
plt.xlabel('Logging Step')
plt.ylabel('Training Loss')
plt.show()

# Ask a question to the model
input_text = "What is the meaning of life?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate a response
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# Decode the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)