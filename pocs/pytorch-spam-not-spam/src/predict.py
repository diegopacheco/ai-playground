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

# Step 2: Create the model
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim)  # New hidden layer
        self.fc2 = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)  # Initialize weights for new layer
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        if text.dim() == 2:
            offsets = None
        embedded = self.embedding(text, offsets)
        out = F.relu(self.fc1(embedded))  # Apply ReLU activation function after first layer
        return self.fc2(out)  # Output layer

# Load the model
device = torch.device("cpu")
EMBED_DIM = 32
NUN_CLASS = 4  # The model was trained with 4 classes
model = TextClassificationModel(len(vocab), EMBED_DIM, NUN_CLASS).to(device)
model.load_state_dict(torch.load('model.pth'))

# Predict function
def predict(text):
    with torch.no_grad():
        text = text_pipeline(text)
        text = torch.tensor(text).unsqueeze(0)  # Add an extra dimension
        output = model(text, None)  # Pass None for offsets
        if output.argmax(1).item() == 1:
            return "SPAM"
        else:
            return "NOT SPAM"

# Test the predict function
print("Win a free trip to Paris! Result: " + predict("Win a free trip to Paris!"))
print("Hello, how are you? Result: " + predict("Hello, how are you?"))
print("Congratulations, you've won a $1000 gift card! Result: " + predict("Congratulations, you've won a $1000 gift card!"))