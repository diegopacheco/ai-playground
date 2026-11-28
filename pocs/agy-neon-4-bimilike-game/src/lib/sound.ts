let audioCtx: AudioContext | null = null;
let soundEnabled = true;

export function setSoundEnabled(enabled: boolean): void {
  soundEnabled = enabled;
}

export function isSoundEnabled(): boolean {
  return soundEnabled;
}

function getContext(): AudioContext | null {
  if (!soundEnabled || typeof window === "undefined") return null;
  const Ctor = window.AudioContext;
  if (!Ctor) return null;
  if (!audioCtx) {
    audioCtx = new Ctor();
  }
  if (audioCtx.state === "suspended") {
    void audioCtx.resume();
  }
  return audioCtx;
}

function playTone(
  freqStart: number,
  freqEnd: number,
  duration: number,
  type: OscillatorType,
  gainPeak: number,
  delay = 0,
): void {
  const ctx = getContext();
  if (!ctx) return;
  const now = ctx.currentTime + delay;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freqStart, now);
  osc.frequency.exponentialRampToValueAtTime(Math.max(40, freqEnd), now + duration);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.linearRampToValueAtTime(gainPeak, now + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(now);
  osc.stop(now + duration + 0.02);
}

export const sound = {
  pop(): void {
    playTone(360, 620, 0.11, "sine", 0.16);
  },
  pickup(): void {
    playTone(420, 740, 0.12, "triangle", 0.15);
  },
  drop(): void {
    playTone(540, 290, 0.13, "sine", 0.16);
  },
  paint(): void {
    playTone(520, 880, 0.1, "sine", 0.14, 0);
    playTone(680, 1080, 0.12, "triangle", 0.12, 0.06);
  },
  bubble(): void {
    playTone(280, 560, 0.09, "sine", 0.14, 0);
    playTone(350, 720, 0.1, "sine", 0.12, 0.07);
  },
  cookSuccess(): void {
    const notes = [523, 659, 784, 1046];
    notes.forEach((n, idx) => {
      playTone(n, n * 1.02, 0.16, "triangle", 0.15, idx * 0.08);
    });
  },
  munch(): void {
    playTone(240, 170, 0.08, "triangle", 0.16, 0);
    playTone(310, 210, 0.08, "triangle", 0.16, 0.1);
    playTone(440, 660, 0.14, "sine", 0.14, 0.2);
  },
  ring(): void {
    playTone(660, 880, 0.14, "sine", 0.13, 0);
    playTone(880, 1100, 0.14, "sine", 0.13, 0.16);
    playTone(660, 880, 0.14, "sine", 0.13, 0.36);
    playTone(880, 1100, 0.14, "sine", 0.13, 0.52);
  },
  unlockFanfare(): void {
    const notes = [523, 659, 784, 988, 1046];
    notes.forEach((n, idx) => {
      playTone(n, n * 1.01, 0.2, "triangle", 0.17, idx * 0.09);
    });
  },
  chirp(pitchMultiplier: number): void {
    const base = 420 * pitchMultiplier;
    playTone(base, base * 1.35, 0.09, "sine", 0.13, 0);
    playTone(base * 1.25, base * 1.55, 0.11, "triangle", 0.12, 0.09);
    playTone(base * 1.1, base * 1.45, 0.1, "sine", 0.11, 0.19);
  },
  speak(text: string, pitch: number, rate: number): void {
    if (!soundEnabled || typeof window === "undefined") return;
    if (!("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.pitch = pitch;
      utter.rate = rate;
      utter.volume = 0.85;
      window.speechSynthesis.speak(utter);
    } catch {
      return;
    }
  },
};
