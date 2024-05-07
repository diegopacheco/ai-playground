# Import necessary libraries
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import requests
import json
import torch

# Initialize Flask app
app = Flask(__name__)

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    chat_output = text_to_chat(message)
    return jsonify({'message': chat_output})

def text_to_chat(initial_text, question):
    # Concatenate the initial text and the question
    text = initial_text + " " + question

    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # Generate a response to the user input
    bot_input_ids = model.generate(new_user_input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id, max_new_tokens=500)

    # Decode the response and return it
    chat_output = tokenizer.decode(bot_input_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return chat_output

def chat_with_bot(initial_text, question):
    result = text_to_chat(initial_text, question)
    return result

ui = gr.Interface(fn=chat_with_bot, 
                     inputs=["text", "text"], 
                     outputs="text",
                     title="Diego's LLM Chat with DialoGPT",
                     examples=
                            [["","What is the capital of France?"],
                            ["A man walked into a bar.", "What happened next?"],
                            ["Acording to NoOne, The purpose of life is about the journey to learn and improve or just 42", "Acording to NoOne, what is the purpose of life?"]],
                     description="DialoGPT is a large-scale pretrained dialogue response generation model. It is based on the GPT-2 architecture with minor modifications to the training objective. The model is trained on multiple tasks, including conversational response generation, and can generate coherent and contextually relevant responses to user inputs.",
                    )
ui.launch()

#if __name__ == '__main__':
    #app.run(debug=True)