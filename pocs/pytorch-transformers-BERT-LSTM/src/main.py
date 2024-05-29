import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = torch.device('cpu')

# Load the IMDB dataset
df = pd.read_csv('data/IMDB_Dataset.csv')

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['review'], df['sentiment'], test_size=0.2)

# Reset the indices of the training and testing sets
train_texts = train_texts.reset_index(drop=True)
test_texts = test_texts.reset_index(drop=True)

# Convert labels to binary
train_labels = [1 if label == 'positive' else 0 for label in train_labels]
test_labels = [1 if label == 'positive' else 0 for label in test_labels]

# Define a custom PyTorch dataset
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        label = torch.tensor(self.labels[idx])
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), label

# Define a custom PyTorch model
class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 256, batch_first=True)
        self.linear = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output, _ = self.lstm(outputs.last_hidden_state)
        output = self.linear(output[:, -1, :])
        return output

# Initialize the model and optimizer
model = SentimentClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters())

# Initialize the data loaders
train_dataset = IMDBDataset(train_texts, train_labels)
test_dataset = IMDBDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Train the model
for epoch in range(10):
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

# Test the model
predictions = []
for input_ids, attention_mask, labels in test_loader:
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    predictions.extend(outputs.squeeze().tolist())

# Plot the predictions
plt.hist(predictions, bins=10)
plt.xlabel('Sentiment score')
plt.ylabel('Number of reviews')
plt.show()