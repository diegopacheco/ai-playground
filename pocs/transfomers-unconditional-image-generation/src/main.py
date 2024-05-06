from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_tensor
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import gradio as gr
from torchvision.utils import save_image
from PIL import Image
import json

example_1 = "ninja turtles fithing against a mosquito, in the sea"

def text_to_comics(text):
    if text == "turtles fithing epic mosquitos in the swear of the city":
        return [Image.open("comics_1.png")]

    pipeline = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")
    output = pipeline(text, prompt_len=70, num_images=1, return_tensors=True)
    
    output_dict = {key: value.tolist() for key, value in output.items()}
    with open("output.json", "w") as f:
        json.dump(output_dict, f)

    images = []
    for i in range(1):
        image = output.images[i]
        image_tensor = to_tensor(image)
        image_tensor = image_tensor / image_tensor.max()
        save_image(image_tensor, f"comics_{i}.png")
        print(f"Image saved as comics_{i}.png")
        images.append(image_tensor)

    return images

e1 = text_to_comics(example_1)

ui = gr.Interface(fn=text_to_comics, 
                  inputs="text", 
                  outputs=["image" for _ in range(4)],
                  title="Type some text get comics!",
                  description="This model generates 3 comics based on the text(max 70 chars) you provide.",
                  examples=[(example_1)],
                 )
ui.launch()