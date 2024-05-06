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
    images = []
    image = Image.open(image_path)
    image_tensor = to_tensor(image)
    image_tensor = image_tensor / image_tensor.max()
    images.append(image_tensor)
    return images

def text_to_comics(text):
    if text == example_1:
        return to_pil_image(load_image("e1_comics_0.png")[0])        
    #if text == example_2:
     #   return to_pil_image(load_image("e2_comics_0.png")[0])
    #if text == example_3:
     #   return to_pil_image(load_image("e3_comics_0.png")[0])

    pipeline = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")
    output = pipeline(text, prompt_len=70, num_images=1, return_tensors=True)

    images = []
    for i in range(1):
        image = output.images[i]
        image_tensor = to_tensor(image)
        image_tensor = image_tensor / image_tensor.max()
        save_image(image_tensor, f"comics_{i}.png")
        images.append(to_pil_image(image_tensor))
    return images[0]

#text_to_comics(example_1)
text_to_comics(example_2)
#text_to_comics(example_3)

ui = gr.Interface(fn=text_to_comics, 
                  inputs="text", 
                  outputs=["image" for _ in range(1)],
                  title="GenAI LLM comics: Type some text get comics!",
                  description="This model generates comics based on the text(max 70 chars) you provide." + \
                    "<BR/>It does not work on mobile(timeout issue) click on examples if dont want to wait. " + \
                    "<BR/> It may take ~10-20min to generate the comics.",
                  examples=[(example_1),(example_2),(example_3)],
                 )
ui.launch()