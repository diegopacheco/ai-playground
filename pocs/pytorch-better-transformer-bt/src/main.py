import torch
import torch.nn as nn

print(f"torch version: {torch.__version__}")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"torch cuda available: {torch.cuda.is_available()}")

import torch, torchtext
from torchtext.models import RobertaClassificationHead
from torchtext.functional import to_tensor
xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim = 1024)
model = xlmr_large.get_model(head=classifier_head)
transform = xlmr_large.transform()

small_input_batch = [
               "Hello world",
               "How are you!"
]
big_input_batch = [
               "Hello world",
               "How are you!",
               """`Well, Prince, so Genoa and Lucca are now just family estates of the
Buonapartes. But I warn you, if you don't tell me that this means war,
if you still try to defend the infamies and horrors perpetrated by
that Antichrist- I really believe he is Antichrist- I will have
nothing more to do with you and you are no longer my friend, no longer
my 'faithful slave,' as you call yourself! But how do you do? I see
I have frightened you- sit down and tell me all the news.`

It was in July, 1805, and the speaker was the well-known Anna
Pavlovna Scherer, maid of honor and favorite of the Empress Marya
Fedorovna. With these words she greeted Prince Vasili Kuragin, a man
of high rank and importance, who was the first to arrive at her
reception. Anna Pavlovna had had a cough for some days. She was, as
she said, suffering from la grippe; grippe being then a new word in
St. Petersburg, used only by the elite."""]

input_batch=big_input_batch
model_input = to_tensor(transform(input_batch), padding_value=1)
output = model(model_input)
output.shape

ITERATIONS=10

print("slow path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
  for i in range(ITERATIONS):
    output = model(model_input)
print(prof)

model.eval()

print("fast path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
  with torch.no_grad():
    for i in range(ITERATIONS):
      output = model(model_input)
print(prof)

model.to(DEVICE)
model_input = model_input.to(DEVICE)

print("slow path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  for i in range(ITERATIONS):
    output = model(model_input)
print(prof)

model.eval()

print("fast path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  with torch.no_grad():
    for i in range(ITERATIONS):
      output = model(model_input)
print(prof)

model.encoder.transformer.layers.enable_nested_tensor = True

model.to(DEVICE)
model_input = model_input.to(DEVICE)

print("slow path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  for i in range(ITERATIONS):
    output = model(model_input)
print(prof)

model.eval()

print("fast path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  with torch.no_grad():
    for i in range(ITERATIONS):
      output = model(model_input)
print(prof)