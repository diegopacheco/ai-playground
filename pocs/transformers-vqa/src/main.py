from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def predict_image_description(image_path, descriptions):
    # Load and process image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)

    # Generate image features
    image_features = model.get_image_features(**inputs)

    # Process descriptions
    inputs = processor(text=descriptions, return_tensors="pt", padding=True)

    # Generate text features
    text_features = model.get_text_features(**inputs)

    # Compute similarity between image and descriptions
    similarity = (text_features @ image_features.T).softmax(dim=-1)

    # Get the description with the highest similarity
    predicted_description = descriptions[similarity.argmax().item()]

    return predicted_description

# Define the image
image_path = "cat_img.jpg"  

print(predict_image_description(image_path, 
  descriptions = ["A cat", "A dog", "A bird", "A car", "A plane", "A person"])
)
