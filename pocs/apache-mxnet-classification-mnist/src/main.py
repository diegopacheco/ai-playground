import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
mnist = mx.test_utils.get_mnist()

# Create a data loader
batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.ArrayDataset(mnist["train_data"], mnist["train_label"]),
    batch_size=batch_size, shuffle=True)

# Define a simple neural network
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128, activation="relu"))
    net.add(nn.Dense(64, activation="relu"))
    net.add(nn.Dense(10))

# Initialize the network
net.initialize(mx.init.Xavier())

# Define loss and trainer
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Train the network
epochs = 10
loss_sequence = []
for epoch in range(epochs):
    cumulative_loss = 0
    for X, y in train_data:
        with autograd.record():
            output = net(X)
            loss = softmax_cross_entropy(output, y)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    print("Epoch %s, loss: %s" % (epoch, cumulative_loss))
    loss_sequence.append(cumulative_loss)

# Plot the decrease in loss over time
plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)
plt.show()