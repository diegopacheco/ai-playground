from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Define the Autoencoder model
input_img = Input(shape=(784,))
encoded = Dense(2, activation='relu')(input_img)  # We reduce to 2 dimensions for easy visualization
decoded = Dense(784, activation='sigmoid')(encoded)

# This model maps an input to its encoded representation
encoder = Model(input_img, encoded)

autoencoder = Model(input_img, decoded)

# Compile and train the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Use the encoder to reduce dimensionality of the test set
encoded_imgs = encoder.predict(x_test)

# Plot the original data
plt.figure(figsize=(8, 6))
plt.scatter(x_test[:, 0], x_test[:, 1], label='Original dimensions')
plt.colorbar()

# Plot the reduced data
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], label='Reduced dimensions')
plt.colorbar()
plt.legend()
plt.show()