from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_tensor
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import gradio as gr
from torchvision.utils import save_image

def text_to_comics(text):
    pipeline = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")
    output = pipeline(text, prompt_len=100, num_images=1, return_tensors=True)

    image = output.images[0]
    image_tensor = to_tensor(image)
    image_tensor = image_tensor / image_tensor.max()
    save_image(image_tensor, "comics.png")
    print("Image saved as comics.png")

    return image_tensor

e1 = text_to_comics("turtles fithing epic mosquitos in the swear of the city")

# ui with gradio
ui = gr.Interface(fn=text_to_comics, 
                  inputs="text", 
                  outputs="image",
                  title="Type some text get comics!",
                  examples=[("turtles fithing epic mosquitos in the swear of the city", e1)],
                 )
ui.launch()