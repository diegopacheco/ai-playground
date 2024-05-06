from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_tensor
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import gradio as gr
from torchvision.utils import save_image
from PIL import Image

def text_to_comics(text):
    if text == "turtles fithing epic mosquitos in the swear of the city":
        image = Image.open("comics.png")
        return [image, image, image, image]

    pipeline = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")
    output = pipeline(text, prompt_len=100, num_images=4, return_tensors=True)

    images = []
    for i in range(4):
        image = output.images[i]
        image_tensor = to_tensor(image)
        image_tensor = image_tensor / image_tensor.max()
        save_image(image_tensor, f"comics_{i}.png")
        print(f"Image saved as comics_{i}.png")
        images.append(image_tensor)

    return images

ui = gr.Interface(fn=text_to_comics, 
                  inputs="text", 
                  outputs=[gr.outputs.Image()]*4,
                  title="Type some text get comics!",
                  examples=[("turtles fithing epic mosquitos in the swear of the city")],
                 )
ui.launch()