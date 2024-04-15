import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Step 1: Prepare data
print("Step 1/4 Data Preparation")

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

# Build the vocab and load the pre-trained word embeddings
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Create the iterators
def text_pipeline(x):
    if isinstance(x, list):
        x = " ".join(map(str, x))
    return vocab(tokenizer(x))

def label_pipeline(x):
    x = int(x)
    return x - 1 if x > 0 else x

# Create a dataset
train_iter = AG_NEWS(split='train')
train_dataset = to_map_style_dataset(train_iter)
train_dataset = [(label_pipeline(label), text_pipeline(text)) for (label, text) in train_dataset]

# Create data loaders
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device), offsets.to(device)

BATCH_SIZE = 64
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# Step 2: Create the model
print("Step 2/4 Model Creation")


# class TextClassificationModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_class):
#         super(TextClassificationModel, self).__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
#         self.fc = nn.Linear(embed_dim, num_class)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero_()

#     def forward(self, text, offsets):
#         if text.dim() == 2:
#             offsets = None
#         embedded = self.embedding(text, offsets)
#         return self.fc(embedded)
# class TextClassificationModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_class):
#         super(TextClassificationModel, self).__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
#         self.fc1 = nn.Linear(embed_dim, embed_dim)  # New hidden layer
#         self.fc2 = nn.Linear(embed_dim, num_class)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc1.weight.data.uniform_(-initrange, initrange)  # Initialize weights for new layer
#         self.fc1.bias.data.zero_()
#         self.fc2.weight.data.uniform_(-initrange, initrange)
#         self.fc2.bias.data.zero_()

#     def forward(self, text, offsets):
#         if text.dim() == 2:
#             offsets = None
#         embedded = self.embedding(text, offsets)
#         out = F.relu(self.fc1(embedded))  # Apply ReLU activation function after first layer
#         return self.fc2(out)  # Output layer
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 2)  # Increase neurons
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)  # New hidden layer
        self.fc3 = nn.Linear(embed_dim, num_class)
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)  # Initialize weights for new layer
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        if text.dim() == 2:
            offsets = None
        embedded = self.embedding(text, offsets)
        out = F.relu(self.fc1(embedded))
        out = self.dropout(out)  # Apply dropout
        out = F.relu(self.fc2(out))  # Apply ReLU activation function after second layer
        out = self.dropout(out)  # Apply dropout
        return self.fc3(out)  # Output layer

# Step 3: Train the model
print("Step 3/4 Model Training")

device = torch.device("cpu")
EMBED_DIM = 32
NUN_CLASS = len(set([label for (label, text) in train_dataset]))
print(f"Number of classes: {NUN_CLASS}")

# num_epochs = 30
# model = TextClassificationModel(len(vocab), EMBED_DIM, NUN_CLASS).to(device)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1.5)  # Reduced learning rate original was 0.4
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

num_epochs = 5  # Increase epochs
model = TextClassificationModel(len(vocab), EMBED_DIM, NUN_CLASS).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2.0)  # Reduced learning rate original was 0.4
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)  # Adjust learning rate decay

total_correct = 0
total_samples = 0
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (label, text, offsets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        output = model(text, offsets)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Adjust learning rate

         # Calculate accuracy
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
        total_samples += label.size(0)
        total_correct += (predicted == label).sum().item()

        # Print training progress
        if i % 500 == 0:  # Print every 500 batches
            accuracy = 100 * correct / total
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}/{len(train_dataloader)}, Loss: {loss.item()}, Accuracy: {accuracy}%")

# Calculate and print total accuracy after all epochs
total_accuracy = 100 * total_correct / total_samples
print(f"Total Accuracy: {total_accuracy}%")

# Save the model after training
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")

# Modify the predict function
print("Step 4/4 Serving Predictions with the model")
def predict(text):
    with torch.no_grad():
        text = text_pipeline(text)
        text = torch.tensor(text).to(device)  # Convert list to tensor
        output = model(text, torch.tensor([0]).to(device))
        if output.argmax(1).item() == 1:
            return "SPAM"
        else:
            return "NOT SPAM"

# Step 4 - predict - Generate 3 sample inputs and test it is spam or not
print("Win a free trip to Paris! Result: " + predict("Win a free trip to Paris!"))
print("Hello, how are you? Result: " + predict("Hello, how are you?"))
print("Congratulations, you've won a $1000 gift card! Result: " + predict("Congratulations, you've won a $1000 gift card!"))