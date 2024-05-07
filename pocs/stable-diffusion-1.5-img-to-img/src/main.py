from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr

model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

def diffuser(prompt, image, strength, guidance_scale):
    images = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale).images
    return images[0]

inputs = [
    gr.Textbox(lines=2, label="Prompt"),
    gr.Image(label="Initial Image"),
    gr.Slider(minimum=0, maximum=1, label="Strength"),
    gr.Slider(minimum=0, maximum=10, label="Guidance Scale")
]

outputs = gr.Image(label="Output Image")
title = "Stable Diffusion"
description = "Generates an image based on the prompt and initial image."
article = ""
examples = [["A fantasy landscape, trending on artstation", "mountains_image.png", 0.75, 7.5]]

gr.Interface(diffuser, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()