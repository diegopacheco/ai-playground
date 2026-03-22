# KittenTTS Fun

Text-to-Speech POC using [KittenTTS](https://github.com/KittenML/KittenTTS) - an open-source TTS library with ultra-compact ONNX models that runs on CPU without GPU.

## What it does

- Generates speech audio from text using KittenTTS (kitten-tts-mini model, 80M params)
- Serves a web UI on port 8080 where you can play the generated audio
- Supports 8 built-in voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- Allows generating new audio from custom text directly in the browser

## Stack

- Python 3.14
- KittenTTS (ONNX-based TTS)
- Flask (web server)
- soundfile (audio I/O)

## How to run

```bash
./run.sh
```

Open http://localhost:8080 in your browser to play audio and generate new speech.

## Result

![](result.png)
