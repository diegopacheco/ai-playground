import torch
import torch.nn as nn

#
# Neutal Network - RNN
# 
# A recurrent neural network (RNN) is a deep learning model that is trained
# to process and convert a sequential data input into a specific sequential
# data output. Sequential data is data—such as words, sentences, or time-series
# data—where sequential components interrelate based on complex semantics and syntax rules.
#
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128