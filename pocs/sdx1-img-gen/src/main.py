import os
import sys

# Get the directory of main.py
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

import gradio as gr
import gradio.helpers
from datasets import load_dataset

import base64
import re
import os
import random
import requests
import time
from PIL import Image
from io import BytesIO
from typing import Tuple

import user_history
from share_btn import community_icon_html, loading_icon_html, share_js

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + negative


#word_list_dataset = load_dataset("google/word-list-sd", data_files="list.txt", use_auth_token=True)
word_list_dataset = load_dataset("ag_news", use_auth_token=True)
word_list = word_list_dataset["train"]['text']

def infer(prompt, negative="low_quality", scale=7, style_name=None, profile: gr.OAuthProfile | None = None):
    for filter in word_list:
        if re.search(rf"\b{re.escape(filter)}\b", prompt):
            raise gr.Error("Please try again with a different prompt")

    seed = random.randint(0,4294967295)
    prompt, negative = apply_style(style_name, prompt, negative)
    images = []
    url = os.getenv('JAX_BACKEND_URL')
    payload = {'instances': [{ 'prompt': prompt, 'negative_prompt': negative, 'parameters':{ 'guidance_scale': scale, 'seed': seed } }] }
    start_time = time.time()
    images_request = requests.post(url, json = payload)
    print(time.time() - start_time)
    try:
        json_data = images_request.json()
    except requests.exceptions.JSONDecodeError:
        raise gr.Error("SDXL did not return a valid result, try again")
    
    for prediction in json_data["predictions"]:
        for image in prediction["images"]:
            image_b64 = (f"data:image/jpeg;base64,{image}")
            images.append(image_b64)

            if profile is not None: # avoid conversion on non-logged-in users
                pil_image = Image.open(BytesIO(base64.b64decode(image)))
                user_history.save_image( # save images + metadata to user history
                    label=prompt,
                    image=pil_image,
                    profile=profile,
                    metadata={
                        "prompt": prompt,
                        "negative_prompt": negative,
                        "guidance_scale": scale,
                    },
                )
    
    return images, gr.update(visible=True)
    
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .gradio-container {
            max-width: 730px !important;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; max-width: 13rem; margin-left: auto;}
        div#share-btn-container > div {flex-direction: row;background: black;align-items: center}
        #share-btn-container:hover {background-color: #060606}
        #share-btn {all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;right:0;}
        #share-btn * {all: unset}
        #share-btn-container div:nth-child(-n+2){width: auto !important;min-height: 0px !important;}
        #share-btn-container .wrap {display: none !important}
        #share-btn-container.hidden {display: none!important}
        
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #prompt-container .form{
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
        }
        #gen-button{
        border-top-left-radius:0;
        border-bottom-left-radius:0;
        }
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
        .tabitem{border: 0 !important}
"""

block = gr.Blocks()

examples = [
    [
        "A serious capybara at work, wearing a suit",
        None,
        None
    ],
    [
        'A Squirtle fine dining with a view to the London Eye',
        None,
        None
    ],
    [
        'A tamale food cart in front of a Japanese Castle',
        None,
        None
    ],
    [
        'a graffiti of a robot serving meals to people',
        None,
        None
    ],
    [
        'a beautiful cabin in Attersee, Austria, 3d animation style',
        None,
        None
    ],
    
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  Fast Stable Diffusion XL on TPU v5e ⚡
                </h1> 
              </div>
              <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                SDXL is a high quality text-to-image model from Stability AI. This demo is running on <a style="text-decoration: underline;" href="https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga">Google Cloud TPU v5e</a>, to achieve efficient and cost-effective inference of 1024×1024 images. <a href="https://hf.co/blog/sdxl_jax" target="_blank">How does it work?</a>
              </p>
            </div>
        """
    )
    
    with gr.Row(elem_id="prompt-container"):
                text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                )    
                btn = gr.Button("Generate", scale=0, elem_id="gen-button")

    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery", columns=2
    )
    
    with gr.Group(elem_id="share-btn-container", visible=False) as community_group:
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")
    
    with gr.Accordion("Advanced settings", open=False):
             style_selection = gr.Radio(
                               show_label=True, container=True, interactive=True,
                               choices=STYLE_NAMES,
                               value=DEFAULT_STYLE_NAME,
                               label='Image Style'
             )
             negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                        elem_id="negative-prompt-text-input",
             )
             guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
             )

    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative, guidance_scale], outputs=[gallery, community_group], cache_examples=True, postprocess=False)
    negative.submit(infer, inputs=[text, negative, guidance_scale, style_selection], outputs=[gallery, community_group], postprocess=False)
    text.submit(infer, inputs=[text, negative, guidance_scale, style_selection], outputs=[gallery, community_group], postprocess=False)
    btn.click(infer, inputs=[text, negative, guidance_scale, style_selection], outputs=[gallery, community_group], postprocess=False)
        
    share_button.click(
            None,
            [],
            [],
            _js=share_js,
    )
    gr.HTML(
            """
                <div class="footer">
                    <p>Model by <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">StabilityAI</a> - backend running JAX on TPUs due to generous support of <a href="https://sites.research.google/trc/about/" style="text-decoration: underline;" target="_blank">Google TRC program</a> - Gradio Demo by 🤗 Hugging Face - this is not an official Google Product
                    </p>
                </div>
           """
    )
    with gr.Accordion(label="License", open=True):
            gr.HTML(
                """<div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md" style="text-decoration: underline;" target="_blank">Stability AI CreativeML Open RAIL++-M</a> license. The License allows users to take advantage of the model in a wide range of settings (including free use and redistribution) as long as they respect the specific use case restrictions outlined, which correspond to model applications the licensor deems ill-suited for the model or are likely to cause harm. For the full list of restrictions please <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. You can read more in the <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
                """
            )

with gr.Blocks(css=css) as block_with_history:
    with gr.Tab("Demo"):
        block.render()
    with gr.Tab("Past generations"):
        user_history.render()

block_with_history.queue(concurrency_count=8, max_size=10, api_open=False).launch(show_api=False)