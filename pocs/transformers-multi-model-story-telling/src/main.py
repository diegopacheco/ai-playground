import os
from PIL import Image
from gtts import gTTS
import torch
import gradio as gr
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

def describe_photo(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
    results = captioner(image)
    text = results[0]['generated_text']
    print(f"Image caption is: {text}")
    return text

def generate_story(description):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tokenizer.encode(description + " [SEP] A funny and friendly story:", return_tensors='pt')
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

def app(image):
    description = describe_photo(image)
    story = generate_story(description)
    audio_file = convert_to_audio(story)
    transcribed_text = audio_to_text(audio_file)
    sentiment = sentiment_analysis(transcribed_text)
    return description,audio_file,transcribed_text, sentiment

ui = gr.Interface(
    fn=app, 
    inputs="image", 
    outputs=["text", "audio", "text", "text"],
    title="Diego's Story Telling Multimodel LLM Gen AI"
)
ui.launch()