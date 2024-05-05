import gradio as gr
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

iface = gr.Interface(fn=generate_image, inputs="text", outputs="image")
iface.launch()