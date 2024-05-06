from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_tensor
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import gradio as gr
from torchvision.utils import save_image
from PIL import Image
import os
from gtts import gTTS
import torch
import gradio as gr
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

example_1 = "ninja turtles fighting against a mosquito, in the sea"
example_2 = "warrior fighting zombies with a sword, in the forest"
example_3 = "western cowboy fighting against a dragon, in the desert"

def load_image(image_path):
    images = []
    image = Image.open(image_path)
    image_tensor = to_tensor(image)
    image_tensor = image_tensor / image_tensor.max()
    images.append(image_tensor)
    return to_pil_image(images[0])

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
        images.append(to_pil_image(image_tensor))
    return images[0]

#text_to_comics(example_1)
#text_to_comics(example_2)
#text_to_comics(example_3)

def generate_story(description):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tokenizer.encode(description + " a thriller/action story.", return_tensors='pt')
    outputs = model.generate(input_ids=inputs, 
                             max_length=200, 
                             num_return_sequences=1, 
                             temperature=0.7, 
                             no_repeat_ngram_size=2)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

def convert_to_audio(text):
    tts = gTTS(text)
    audio_file_path = "audio.mp3"
    tts.save(audio_file_path)
    return audio_file_path

def audio_to_text(audio_file_path):
    pipe = pipeline("automatic-speech-recognition", "openai/whisper-large-v2")
    result = pipe("audio.mp3")
    print(result)
    return result['text']

def sentiment_analysis(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    result = sentiment_analyzer(text)
    print(result)
    return result

def app(text):
    comics = text_to_comics(text)
    story = generate_story(text)
    audio_file = convert_to_audio(story)
    transcribed_text = audio_to_text(audio_file)
    sentiment = sentiment_analysis(transcribed_text)
    return comics, audio_file,transcribed_text, sentiment

ui = gr.Interface(fn=app, 
                  inputs="text", 
                  outputs=["image", "audio", "text", "text"],
                  title="GenAI Multi-model LLM comics: Type some text get comics!",
                  description="This model generates comics based on the text(max 70 chars) you provide." + \
                    "<BR/>It does not work on mobile(timeout issue) click on examples if dont want to wait. " + \
                    "<BR/>It may take ~10-20min to generate the comics.",
                  examples=[(example_1),(example_2),(example_3)],
                 )
ui.launch()