import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Load pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define your data
data = [
    "User click on button 1; user click on button 2; user apply,not suspecious",
    "User dont click on button 1; user dont click on button 2; user dont apply,not suspecious",
    "User click on button 1; user dont click on button 2; user apply,suspecious",
    "User click on button 1; user dont click on button 2; user apply,suspecious",
]

# Preprocess the data
events = [item.split(",")[0] for item in data]
labels = [item.split(",")[1] for item in data]

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Tokenize events
inputs = tokenizer(events, return_tensors="pt", truncation=True, padding=True)

# Define a dataset
class EventDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = EventDataset(inputs, labels)

# Define a trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

def detect_red_flags(event):
    # Tokenize event
    inputs = tokenizer(event, return_tensors="pt", truncation=True, padding=True)
    
    # Make prediction
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1)
    
    # Decode prediction
    prediction = le.inverse_transform(prediction.detach().numpy())
    
    return prediction

# Test the function
event = "User click on button 1; user dont click on button 2; user apply"
prediction = detect_red_flags(event)
print(prediction)