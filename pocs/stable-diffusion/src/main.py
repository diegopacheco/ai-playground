import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

prompt = "a photo of an black astronaut riding a pig on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut.png")
print("Image saved as astronaut.png")