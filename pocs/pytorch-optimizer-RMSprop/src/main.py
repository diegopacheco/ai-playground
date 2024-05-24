import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import os
os.environ['QT_QPA_PLATFORM'] = 'wayland'

# Load the MNIST dataset
dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = dataset['train'], dataset['test']

# Convert the tensorflow dataset to pytorch dataset
train = [(torch.from_numpy(img), torch.tensor(lbl)) for img, lbl in tfds.as_dataframe(mnist_train, info).values]

trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.01)

# Define a list to store the losses
losses = []

# Train the network
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Record the average loss for this epoch
    average_loss = running_loss / len(trainloader)
    losses.append(average_loss)
    print('Epoch %d average loss: %.3f' % (epoch + 1, average_loss))

print('Finished Training')

# Plot the training loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
print('Training loss plot saved as training_loss.png')