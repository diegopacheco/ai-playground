from flask import Flask, send_file, render_template_string
from kittentts import KittenTTS
import os

app = Flask(__name__)

AUDIO_FILE = "output.wav"

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KittenTTS Player</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            backdrop-filter: blur(10px);
            width: 500px;
        }
        h1 {
            font-size: 2em;
            margin-bottom: 10px;
            color: #e94560;
        }
        .subtitle {
            color: #888;
            margin-bottom: 30px;
            font-size: 0.95em;
        }
        .text-box {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 25px;
            font-style: italic;
            color: #ccc;
            line-height: 1.6;
        }
        .voice-info {
            color: #e94560;
            font-size: 0.85em;
            margin-bottom: 20px;
        }
        audio {
            width: 100%;
            margin-bottom: 15px;
        }
        .generate-form {
            margin-top: 20px;
        }
        textarea {
            width: 100%;
            height: 80px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            color: #e0e0e0;
            padding: 12px;
            font-size: 0.95em;
            resize: vertical;
            margin-bottom: 10px;
        }
        select {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            color: #e0e0e0;
            padding: 8px 12px;
            font-size: 0.9em;
            margin-bottom: 10px;
            width: 100%;
        }
        option { background: #1a1a2e; }
        button {
            background: #e94560;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            width: 100%;
            transition: background 0.2s;
        }
        button:hover { background: #c73652; }
        button:disabled { background: #666; cursor: wait; }
        .status { color: #888; font-size: 0.85em; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>KittenTTS</h1>
        <p class="subtitle">Text-to-Speech with KittenML</p>

        <div id="player-section">
            <audio id="audio-player" controls autoplay>
                <source src="/audio" type="audio/wav">
            </audio>
        </div>

        <div class="generate-form">
            <textarea id="text-input" placeholder="Type text to generate speech...">Artificial intelligence is transforming the way we interact with technology every single day.</textarea>
            <select id="voice-select">
                <option value="Jasper">Jasper</option>
                <option value="Bella">Bella</option>
                <option value="Luna">Luna</option>
                <option value="Bruno">Bruno</option>
                <option value="Rosie">Rosie</option>
                <option value="Hugo">Hugo</option>
                <option value="Kiki">Kiki</option>
                <option value="Leo">Leo</option>
            </select>
            <button id="gen-btn" onclick="generate()">Generate Speech</button>
            <p class="status" id="status"></p>
        </div>
    </div>
    <script>
        async function generate() {
            const btn = document.getElementById('gen-btn');
            const status = document.getElementById('status');
            const text = document.getElementById('text-input').value;
            const voice = document.getElementById('voice-select').value;

            btn.disabled = true;
            status.textContent = 'Generating audio...';

            try {
                const resp = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, voice: voice})
                });
                if (resp.ok) {
                    const player = document.getElementById('audio-player');
                    player.src = '/audio?t=' + Date.now();
                    player.load();
                    player.play();
                    status.textContent = 'Done!';
                } else {
                    status.textContent = 'Error generating audio.';
                }
            } catch(e) {
                status.textContent = 'Error: ' + e.message;
            }
            btn.disabled = false;
        }
    </script>
</body>
</html>
"""

model = None

def get_model():
    global model
    if model is None:
        model = KittenTTS("KittenML/kitten-tts-mini-0.8")
    return model

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/audio")
def audio():
    if os.path.exists(AUDIO_FILE):
        return send_file(AUDIO_FILE, mimetype="audio/wav")
    return "No audio file found", 404

@app.route("/generate", methods=["POST"])
def generate():
    from flask import request
    data = request.get_json()
    text = data.get("text", "Hello world")
    voice = data.get("voice", "Jasper")
    m = get_model()
    m.generate_to_file(text, AUDIO_FILE, voice=voice, speed=1.0)
    return {"status": "ok"}

if __name__ == "__main__":
    print("Generating initial audio...")
    m = get_model()
    m.generate_to_file(
        "Artificial intelligence is transforming the way we interact with technology every single day.",
        AUDIO_FILE,
        voice="Jasper",
        speed=1.0
    )
    print(f"Audio saved to {AUDIO_FILE}")
    print("Starting server at http://localhost:8080")
    app.run(host="0.0.0.0", port=8080)
