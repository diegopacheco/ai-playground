import torch
from torch import nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the California housing dataset
data = fetch_california_housing()
X = data['data']
y = data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Define the neural network
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    # Forward pass
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test the model
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()

# Prediction function
def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).squeeze()
    return predictions

# Convert tensors to numpy arrays for plotting
y_test_np = y_test.numpy()
predictions_np = predictions.numpy()

# Calculate the baseline predictions (average house price)
baseline_predictions = np.full(y_test_np.shape, y_test_np.mean())

# Plot the true prices
plt.scatter(range(len(y_test_np)), y_test_np, color='blue', label='True Prices')

# Plot the predicted prices
plt.scatter(range(len(predictions_np)), predictions_np, color='red', label='Predicted Prices')

# Plot the baseline
plt.plot(range(len(baseline_predictions)), baseline_predictions, color='green', label='Baseline')

plt.xlabel("House")
plt.ylabel("Price")
plt.title("True vs Predicted House Prices")
plt.legend()
plt.show()