import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
import os
from threading import Thread

token = os.environ["HF_TOKEN"]
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    token=token,
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", token=token)
terminators = [
    tok.eos_token_id,
]
device = torch.device("cpu")
model = model.to(device)

def chat(message, history, temperature, do_sample, max_tokens):
    chat = []
    for item in history:
        chat.append({"role": "user", "content": item[0]})
        if item[1] is not None:
            chat.append({"role": "assistant", "content": item[1]})
    chat.append({"role": "user", "content": message})
    messages = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    model_inputs = tok([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tok, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )

    if temperature == 0:
        generate_kwargs["do_sample"] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text
    yield partial_text

ui = gr.ChatInterface(
    fn=chat,
    examples=[["write a guessing game in python"],["Write a hello world program in rust"],["Write a calculator class in scala and some unit tests"]],
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.9, label="Temperature", render=False
        ),
        gr.Checkbox(label="Sampling", value=True),
        gr.Slider(
            minimum=128,
            maximum=4096,
            step=1,
            value=512,
            label="Max new tokens",
            render=False,
        ),
    ],
    stop_btn="Stop Generation",
    title="Chat With LLMs",
    description="LLM Chat using microsoft/Phi-3-mini-128k-instruct model.",
).launch()