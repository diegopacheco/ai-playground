from transformers import pipeline
from PIL import Image
import gradio as gr
import numpy as np

estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")

def depth_estimation(image_url):
    image = Image.fromarray((image_url * 255).astype(np.uint8))
    result = estimator(images=image)
    return result['depth']

gr.Interface(fn=depth_estimation, inputs="image", outputs="image").launch()