import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the Stacked Denoising Autoencoder (SDAE)
class SDAE(nn.Module):
    def __init__(self):
        super(SDAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 28 * 28), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize the model, criterion, and optimizer
model = SDAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for data in trainloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img_noisy = img + 0.5 * torch.randn(*img.shape)
        img_noisy = torch.clamp(img_noisy, 0., 1.)
        output = model(img_noisy)
        loss = criterion(output, img.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test the model
img = img.view(img.size(0), 1, 28, 28)
img_noisy = img_noisy.view(img_noisy.size(0), 1, 28, 28)
output = output.view(output.size(0), 1, 28, 28)

# Plot the results
fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25,6))
for images, row in zip([img, img_noisy, output.detach()], axes):
    for img, ax in zip(images, row):
        ax.imshow(torch.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()