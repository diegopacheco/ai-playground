import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)
from torch.utils.data import DataLoader
from torchtext.datasets import SST2

batch_size = 16
train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")
print("1. Train dataset length = [{}], Dev dataset length = [{}]".format(train_datapipe,dev_datapipe))

# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
def apply_transform(x):
    return text_transform(x[0]), x[1]

train_datapipe = train_datapipe.map(apply_transform)
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)
dev_datapipe = dev_datapipe.map(apply_transform)
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)
num_classes = 2
input_dim = 768

from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)

import torchtext.functional as F
from torch.optim import AdamW

learning_rate = 1e-5
optim = AdamW(model.parameters(), lr=learning_rate)
criteria = nn.CrossEntropyLoss()
print("2. Model = [{}], Optimizer = [{}], Criteria = [{}]".format(model, optim, criteria))

def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1
            print("3. Counter = [{}], loss = [{}], correct_predictions = [{}], total_predictions = [{}]".format(counter, loss, correct_predictions, total_predictions))

    return total_loss / counter, correct_predictions / total_predictions

num_epochs = 1
for e in range(num_epochs):
    counter = 0
    for batch in train_dataloader:
        input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
        target = torch.tensor(batch["target"]).to(DEVICE)
        train_step(input, target)
        counter += 1
        print("Counter = [{}]".format(counter))

    loss, accuracy = evaluate()
    print("4. Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))

# predict
def predict(text):
    model.eval()
    with torch.no_grad():
        input = text_transform(text)
        input = F.to_tensor([input], padding_value=padding_idx).to(DEVICE)
        output = model(input)
        return output.argmax(1).item()

print("5. This film is terrible: ")    
print(predict("This film is terrible"))
