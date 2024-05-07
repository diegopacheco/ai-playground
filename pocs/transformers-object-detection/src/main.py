from PIL import Image
import requests
from transformers import ViTFeatureExtractor, ViTForImageClassification
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr
import numpy as np

def classify_and_label_image(image_array):
    # Convert numpy array to PIL Image
    image = Image.fromarray(np.uint8(image_array))

    # Load pre-trained model and feature extractor
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Make prediction
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Get label of the predicted class
    labels = model.config.id2label
    predicted_class_label = labels[predicted_class_idx]

    # Calculate prediction probability
    probabilities = logits.softmax(dim=-1)
    predicted_class_prob = probabilities[0, predicted_class_idx].item()

    # Draw bounding box and label on the image
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle((50, 50), 100, 100, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(50, 40, f'{predicted_class_label} {predicted_class_prob * 100:.2f}%', color='r')

    # Convert the figure to a numpy array
    fig.canvas.draw()
    img_arr = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the figure to free up memory
    plt.close(fig)

    # Return the image array
    return img_arr

examples = [
    ["bear.jpg"],
    ["puppy.jpg"],
    ["boat.jpg"]
]
gr.Interface(fn=classify_and_label_image, title="Diego's Image to Labeled Image",
                                          description="Classify an image and draw the label on the image.",
                                          examples=examples,
                                          inputs="image",
                                          outputs="image")\
.launch(share=False, server_port=8080)