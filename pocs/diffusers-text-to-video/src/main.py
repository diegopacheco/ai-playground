import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import gradio as gr

os.environ["CUDA_VISIBLE_DEVICES"] = ""

mdevice = torch.device('cpu')
pipe = DiffusionPipeline.from_pretrained(device=mdevice, pretrained_model_name_or_path="damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, device=mdevice)

def text_to_video(prompt):
    video_frames = pipe(prompt, num_inference_steps=25).frames
    # Ensure video_frames[0] is a 3D tensor
    if len(video_frames[0].shape) == 4:
        # Select the first frame from the batch
        frame = video_frames[0][0]
    else:
        frame = video_frames[0]

    video_path = export_to_video([frame])
    return video_path

gr.Interface(fn=text_to_video, inputs="text", outputs="file").launch()