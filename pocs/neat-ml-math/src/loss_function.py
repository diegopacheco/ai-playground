import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# True and predicted values
y_true = np.linspace(0, 10, 100)

# Call the mse_loss function with different predicted values
y_pred1 = y_true + np.random.normal(0, 0.5, size=y_true.shape)  # Add some noise
loss1 = mse_loss(y_true, y_pred1)

y_pred2 = y_true + np.random.normal(0, 1, size=y_true.shape)  # Add more noise
loss2 = mse_loss(y_true, y_pred2)

y_pred3 = y_true + np.random.normal(0, 1.5, size=y_true.shape)  # Add even more noise
loss3 = mse_loss(y_true, y_pred3)

# Plot the results
plt.figure()
plt.plot(y_true, y_pred1, label=f'Loss: {loss1:.2f}')
plt.plot(y_true, y_pred2, label=f'Loss: {loss2:.2f}')
plt.plot(y_true, y_pred3, label=f'Loss: {loss3:.2f}')
plt.title('MSE loss for different predictions')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()