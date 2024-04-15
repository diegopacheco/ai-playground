import torch
from torchvision import transforms
from PIL import Image
from torch import nn
from torchvision import models
import numpy as np

# Load the model
model = models.densenet121(pretrained=False)
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))
model.load_state_dict(torch.load('hotdog_not_hotdog.pth'))
model.eval()

def predict(image_path):
    # Define transformations for the image
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Open the image file
    img = Image.open(image_path)

    # Apply the transformations to the image
    image_tensor = transform(img)

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    # Set the model to evaluation mode
    model.eval()

    # Predict the class of the image
    with torch.no_grad():
        output = model(image_tensor)

    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())

    if preds == 0:
        return "Hotdog"
    else:
        return "Not hotdog"

print("data/train/hotdog/106.jpg             -> " + predict("data/train/hotdog/106.jpg"))
print("data/train/hotdog/120.jpg             -> " + predict("data/train/hotdog/120.jpg"))
print("data/train/nothotdog/101.jpg          -> " + predict("data/train/nothotdog/101.jpg"))

print("data/predict-test/blue-car.jpg        -> " + predict("data/predict-test/blue-car.jpg"))
print("data/predict-test/hotdog.jpg          -> " + predict("data/predict-test/hotdog.jpg"))
print("data/predict-test/doudle-hotdog.jpg   -> " + predict("data/predict-test/doudle-hotdog.jpg"))
print("data/predict-test/pet-dog.jpg         -> " + predict("data/predict-test/pet-dog.jpg"))
print("data/predict-test/pet-dog2.jpg        -> " + predict("data/predict-test/pet-dog2.jpg"))