import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size) #hidden_size)
        self.i2o = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size) #output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        print(f"input {input.size()}")
        print(f"hidden {hidden.size()}")

        combined = torch.cat((input, hidden), 1)
        print(f"Combined {combined.size()}")

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))