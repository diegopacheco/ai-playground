from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_tensor
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import gradio as gr
from torchvision.utils import save_image
from PIL import Image

example_1 = "ninja turtles fithing against a mosquito, in the sea"
example_2 = "warrior fithing zombies with a sword, in the forest"
example_3 = "western cowboy fithing against a dragon, in the desert"

def load_image(image_path):
    image = Image.open(image_path)
    image_tensor = to_tensor(image)
    image_tensor = image_tensor / image_tensor.max()
    return [image_tensor]

def text_to_comics(text):
    if text == example_1:
        return load_image("e1_comics_0.png")        
    if text == example_2:
        return load_image("e2_comics_0.png")
    if text == example_3:
        return load_image("e3_comics_0.png")

    pipeline = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")
    output = pipeline(text, prompt_len=70, num_images=1, return_tensors=True)

    images = []
    for i in range(1):
        image = output.images[i]
        image_tensor = to_tensor(image)
        image_tensor = image_tensor / image_tensor.max()
        save_image(image_tensor, f"comics_{i}.png")
        print(f"Image saved as comics_{i}.png")
        images.append(image_tensor)

    return images

#text_to_comics(example_1)
#text_to_comics(example_2)
#text_to_comics(example_3)

ui = gr.Interface(fn=text_to_comics, 
                  inputs="text", 
                  outputs="image",
                  title="Type some text get comics!",
                  description="This model generates comics based on the text(max 70 chars) you provide.",
                  examples=[(example_1),(example_2),(example_3)],
                 )
ui.launch()