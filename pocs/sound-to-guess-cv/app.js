import { AudioClassifier, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.20";

const els = {
  toggle: document.getElementById("toggle"),
  status: document.getElementById("status"),
  emoji: document.getElementById("emoji"),
  label: document.getElementById("label"),
  meter: document.getElementById("meter"),
  confidence: document.getElementById("confidence"),
  topList: document.getElementById("topList"),
  bars: document.getElementById("bars"),
  halo: document.getElementById("halo")
};

const EMOJI = [
  [["clap", "applause"], "👏"],
  [["finger snap", "snap"], "🫰"],
  [["chicken", "rooster", "cluck", "fowl", "crow", "cock-a-doodle"], "🐔"],
  [["door", "knock", "slam", "doorbell"], "🚪"],
  [["whistl"], "😗"],
  [["speech", "conversation", "narration", "babbling", "monologue", "shout", "yell"], "🗣️"],
  [["sing", "song", "music", "musical", "guitar", "piano", "drum"], "🎵"],
  [["dog", "bark", "bow-wow", "howl", "growl"], "🐶"],
  [["cat", "meow", "purr"], "🐱"],
  [["laugh", "giggle", "chuckle", "snicker"], "😂"],
  [["cough", "sneeze", "throat", "sniff"], "🤧"],
  [["alarm", "bell", "ring", "siren", "beep"], "🔔"],
  [["water", "drip", "splash", "pour", "liquid"], "💧"],
  [["wind", "rustl"], "🌬️"],
  [["typ", "keyboard", "click"], "⌨️"],
  [["footstep", "walk", "run"], "👣"],
  [["breath", "sigh"], "😮‍💨"],
  [["silence", "quiet"], "🤫"],
  [["hum", "vibration", "buzz", "sine"], "🎚️"]
];

const emojiFor = (name) => {
  const n = name.toLowerCase();
  for (const [keys, e] of EMOJI) if (keys.some((k) => n.includes(k))) return e;
  return "🔊";
};

let classifier = null;
let audioContext = null;
let stream = null;
let source = null;
let processor = null;
let mute = null;
let timer = null;
let running = false;

let ring = null;
let writePos = 0;
let filled = false;
let windowSize = 0;
let sampleRate = 16000;
let lastEmoji = "";

async function loadModel() {
  try {
    const fileset = await FilesetResolver.forAudioTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.20/wasm"
    );
    classifier = await AudioClassifier.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
      },
      maxResults: 4
    });
    els.toggle.disabled = false;
    els.toggle.textContent = "Start Listening";
    els.status.textContent = "Model ready. Click start and allow your microphone.";
  } catch (err) {
    els.status.textContent = "Could not load the model: " + err.message;
  }
}

function setLevel(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / samples.length);
  const pct = Math.min(100, Math.round(rms * 320));
  els.meter.style.width = pct + "%";
}

function classifyNow() {
  if (!classifier) return;
  if (!filled && writePos < windowSize) return;
  const buf = new Float32Array(windowSize);
  const start = (writePos - windowSize + ring.length) % ring.length;
  for (let i = 0; i < windowSize; i++) buf[i] = ring[(start + i) % ring.length];

  const out = classifier.classify(buf, sampleRate);
  const cats = out?.[0]?.classifications?.[0]?.categories || [];
  if (cats.length) render(cats);
}

function render(cats) {
  const top = cats[0];
  const emoji = emojiFor(top.categoryName);
  const pct = Math.round(top.score * 100);

  els.emoji.textContent = emoji;
  els.label.textContent = top.categoryName;
  els.confidence.textContent = pct + "% sure";

  if (emoji !== lastEmoji) {
    lastEmoji = emoji;
    els.halo.classList.remove("pop");
    void els.halo.offsetWidth;
    els.halo.classList.add("pop");
  }

  els.topList.innerHTML = cats
    .slice(0, 4)
    .map((c) => {
      const w = Math.round(c.score * 100);
      return `<li>
        <span class="t-emoji">${emojiFor(c.categoryName)}</span>
        <span class="t-name">${c.categoryName}</span>
        <span class="t-score">${w}%</span>
        <span class="t-bar"><i style="width:${w}%"></i></span>
      </li>`;
    })
    .join("");
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    els.status.textContent = "Microphone blocked: " + err.message;
    return;
  }

  audioContext = new AudioContext();
  sampleRate = audioContext.sampleRate;
  windowSize = Math.floor(sampleRate * 0.975);
  ring = new Float32Array(sampleRate);
  writePos = 0;
  filled = false;

  source = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  mute = audioContext.createGain();
  mute.gain.value = 0;

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    for (let i = 0; i < input.length; i++) {
      ring[writePos] = input[i];
      writePos = (writePos + 1) % ring.length;
      if (writePos === 0) filled = true;
    }
    setLevel(input);
  };

  source.connect(processor);
  processor.connect(mute);
  mute.connect(audioContext.destination);

  timer = setInterval(classifyNow, 280);
  running = true;

  els.toggle.textContent = "Stop";
  els.toggle.classList.add("listening");
  els.bars.dataset.active = "true";
  els.halo.classList.add("live");
  els.status.textContent = "Listening… make a sound!";
  els.label.textContent = "Listening…";
  els.emoji.textContent = "👂";
}

function stop() {
  running = false;
  if (timer) clearInterval(timer);
  timer = null;
  if (processor) processor.onaudioprocess = null;
  try { source && source.disconnect(); } catch (e) {}
  try { processor && processor.disconnect(); } catch (e) {}
  try { mute && mute.disconnect(); } catch (e) {}
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (audioContext) audioContext.close();
  audioContext = null;
  stream = null;

  els.toggle.textContent = "Start Listening";
  els.toggle.classList.remove("listening");
  els.bars.dataset.active = "false";
  els.halo.classList.remove("live");
  els.meter.style.width = "0%";
  els.status.textContent = "Stopped. Click start to listen again.";
}

els.toggle.addEventListener("click", () => {
  if (running) stop();
  else start();
});

loadModel();
