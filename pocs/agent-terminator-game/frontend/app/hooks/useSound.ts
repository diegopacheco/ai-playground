import { useCallback, useRef, useState } from "react";

type SoundName = "kill" | "date" | "hatch" | "start" | "victory" | "swarm";

function createAudioContext(): AudioContext {
  return new (window.AudioContext || (window as any).webkitAudioContext)();
}

function playKillSound(ctx: AudioContext) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(300, ctx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(80, ctx.currentTime + 0.3);
  gain.gain.setValueAtTime(0.4, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4);
  osc.start(ctx.currentTime);
  osc.stop(ctx.currentTime + 0.4);

  const osc2 = ctx.createOscillator();
  const gain2 = ctx.createGain();
  osc2.connect(gain2);
  gain2.connect(ctx.destination);
  osc2.type = "square";
  osc2.frequency.setValueAtTime(150, ctx.currentTime + 0.1);
  osc2.frequency.exponentialRampToValueAtTime(40, ctx.currentTime + 0.5);
  gain2.gain.setValueAtTime(0.3, ctx.currentTime + 0.1);
  gain2.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
  osc2.start(ctx.currentTime + 0.1);
  osc2.stop(ctx.currentTime + 0.5);

  const bufferSize = ctx.sampleRate * 0.2;
  const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
  }
  const noise = ctx.createBufferSource();
  const noiseGain = ctx.createGain();
  noise.buffer = buffer;
  noise.connect(noiseGain);
  noiseGain.connect(ctx.destination);
  noiseGain.gain.setValueAtTime(0.3, ctx.currentTime);
  noiseGain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
  noise.start(ctx.currentTime);
}

function playDateSound(ctx: AudioContext) {
  const notes = [523, 659, 784, 880];
  notes.forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.12);
    gain.gain.setValueAtTime(0, ctx.currentTime + i * 0.12);
    gain.gain.linearRampToValueAtTime(0.25, ctx.currentTime + i * 0.12 + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + i * 0.12 + 0.2);
    osc.start(ctx.currentTime + i * 0.12);
    osc.stop(ctx.currentTime + i * 0.12 + 0.25);
  });
}

function playHatchSound(ctx: AudioContext) {
  for (let i = 0; i < 3; i++) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sine";
    const t = ctx.currentTime + i * 0.08;
    osc.frequency.setValueAtTime(800 + i * 400, t);
    osc.frequency.exponentialRampToValueAtTime(2000 + i * 300, t + 0.06);
    gain.gain.setValueAtTime(0.2, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + 0.1);
    osc.start(t);
    osc.stop(t + 0.1);
  }

  const bufferSize = ctx.sampleRate * 0.08;
  const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize) * 0.5;
  }
  const noise = ctx.createBufferSource();
  const noiseGain = ctx.createGain();
  noise.buffer = buffer;
  noise.connect(noiseGain);
  noiseGain.connect(ctx.destination);
  noiseGain.gain.setValueAtTime(0.15, ctx.currentTime);
  noiseGain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
  noise.start(ctx.currentTime + 0.05);
}

function playStartSound(ctx: AudioContext) {
  const notes = [196, 247, 294, 392];
  notes.forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.15);
    gain.gain.setValueAtTime(0.3, ctx.currentTime + i * 0.15);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + i * 0.15 + 0.3);
    osc.start(ctx.currentTime + i * 0.15);
    osc.stop(ctx.currentTime + i * 0.15 + 0.3);
  });
}

function playVictorySound(ctx: AudioContext) {
  const notes = [392, 494, 587, 784, 784, 587, 784];
  const durations = [0.15, 0.15, 0.15, 0.3, 0.1, 0.1, 0.4];
  let t = ctx.currentTime;
  notes.forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "square";
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.25, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + durations[i]);
    osc.start(t);
    osc.stop(t + durations[i] + 0.05);
    t += durations[i];
  });
}

function playSwarmSound(ctx: AudioContext) {
  for (let i = 0; i < 5; i++) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sawtooth";
    const baseFreq = 100 + Math.random() * 80;
    osc.frequency.setValueAtTime(baseFreq, ctx.currentTime);
    const lfo = ctx.createOscillator();
    const lfoGain = ctx.createGain();
    lfo.connect(lfoGain);
    lfoGain.connect(osc.frequency);
    lfo.frequency.setValueAtTime(20 + Math.random() * 30, ctx.currentTime);
    lfoGain.gain.setValueAtTime(50, ctx.currentTime);
    lfo.start(ctx.currentTime);
    lfo.stop(ctx.currentTime + 1.5);
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.08, ctx.currentTime + 0.3);
    gain.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.8);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 1.5);
    osc.start(ctx.currentTime + i * 0.1);
    osc.stop(ctx.currentTime + 1.5);
  }
}

const SOUND_PLAYERS: Record<SoundName, (ctx: AudioContext) => void> = {
  kill: playKillSound,
  date: playDateSound,
  hatch: playHatchSound,
  start: playStartSound,
  victory: playVictorySound,
  swarm: playSwarmSound,
};

export function useSound() {
  const [muted, setMuted] = useState(false);
  const ctxRef = useRef<AudioContext | null>(null);

  const play = useCallback(
    (name: SoundName) => {
      if (muted) return;
      try {
        if (!ctxRef.current) {
          ctxRef.current = createAudioContext();
        }
        if (ctxRef.current.state === "suspended") {
          ctxRef.current.resume();
        }
        SOUND_PLAYERS[name](ctxRef.current);
      } catch {}
    },
    [muted]
  );

  const toggleMute = useCallback(() => {
    setMuted((prev) => !prev);
  }, []);

  return { play, muted, toggleMute };
}
