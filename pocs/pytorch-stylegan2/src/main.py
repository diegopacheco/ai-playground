import torch
import numpy as np
import dnnlib
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader

def generate_synthetic_images(num_images):
    # Load the pre-trained StyleGAN2 model
    model = ModelLoader().load("stylegan2-ffhq-config-f.pt")
    model.eval()

    # Generate random noise
    z = torch.randn(num_images, 512).to('gpu') # move input to cpu

    # Generate synthetic images
    with torch.no_grad():
        images = model(z, None, truncation_psi=0.5)

    # Save the images
    for i, image in enumerate(images):
        save_image(image.add(1).div(2), f'synthetic_image_{i}.png') # normalize image to [0,1] range

# Generate 5 synthetic images
generate_synthetic_images(5)