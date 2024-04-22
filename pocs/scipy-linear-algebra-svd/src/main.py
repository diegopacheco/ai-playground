import numpy as np
from scipy.linalg import svd
import imageio
import matplotlib.pyplot as plt

# Load the image
img = imageio.imread('image.jpg')

# Convert the image to grayscale
gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# Perform SVD on the grayscale image
U, s, Vh = svd(gray_img, full_matrices=False)

# Keep only the top k singular values and the corresponding singular vectors
k = 50
U_k = U[:, :k]
s_k = s[:k]
Vh_k = Vh[:k, :]

# Reconstruct the compressed image
compressed_img = np.dot(U_k, np.dot(np.diag(s_k), Vh_k))

# Normalize the image to 0-255
compressed_img = (compressed_img - compressed_img.min()) / (compressed_img.max() - compressed_img.min())
compressed_img = (compressed_img * 255).astype(np.uint8)

# Save the compressed image
imageio.imsave('compressed_image.jpg', compressed_img)

# Save the compressed image
imageio.imsave('compressed_image.jpg', compressed_img)

# Display the original and compressed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title('Compressed Image')
plt.show()