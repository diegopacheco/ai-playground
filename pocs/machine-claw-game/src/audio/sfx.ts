export type Sfx = {
  unlock(): void;
  coin(): void;
  start(): void;
  drop(): void;
  grab(): void;
  win(): void;
  miss(): void;
  motor(active: boolean): void;
  dispose(): void;
};

type Note = readonly [frequency: number, delay: number, length: number];
type Motor = { readonly gain: GainNode; readonly voices: readonly OscillatorNode[] };

const MOTOR_PITCH: readonly number[] = [196, 294, 392];
const MOTOR_LEVEL: readonly number[] = [0.5, 0.3, 0.12];

export function createSfx(): Sfx {
  let context: AudioContext | undefined;
  let output: AudioNode | undefined;
  let motor: Motor | undefined;

  function buildMotor(audio: AudioContext, destination: AudioNode): Motor {
    const gain = audio.createGain();
    const filter = audio.createBiquadFilter();
    const purr = audio.createOscillator();
    const purrDepth = audio.createGain();
    gain.gain.value = 0;
    filter.type = "lowpass";
    filter.frequency.value = 900;
    filter.Q.value = 0.4;
    purr.frequency.value = 11;
    purrDepth.gain.value = 2.5;
    purr.connect(purrDepth);
    purr.start();
    const voices = MOTOR_PITCH.map((pitch, index) => {
      const voice = audio.createOscillator();
      const level = audio.createGain();
      voice.type = "sine";
      voice.frequency.value = pitch * 0.8;
      level.gain.value = MOTOR_LEVEL[index] ?? 0;
      purrDepth.connect(voice.frequency);
      voice.connect(level).connect(filter);
      voice.start();
      return voice;
    });
    filter.connect(gain).connect(destination);
    return { gain, voices };
  }

  function unlock(): void {
    if (context) {
      void context.resume();
      return;
    }
    context = new AudioContext();
    const soften = context.createBiquadFilter();
    soften.type = "lowpass";
    soften.frequency.value = 4200;
    soften.connect(context.destination);
    output = soften;
    motor = buildMotor(context, soften);
  }

  function play(type: OscillatorType, volume: number, notes: readonly Note[]): void {
    if (!context || !output) return;
    for (const [frequency, delay, length] of notes) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const at = context.currentTime + delay;
      oscillator.type = type;
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, at);
      gain.gain.exponentialRampToValueAtTime(volume, at + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + length);
      oscillator.connect(gain).connect(output);
      oscillator.start(at);
      oscillator.stop(at + length + 0.02);
    }
  }

  return {
    unlock,
    coin: () => play("sine", 0.22, [[1320, 0, 0.1], [1980, 0.07, 0.4]]),
    start: () => play("triangle", 0.12, [[523, 0, 0.12], [659, 0.1, 0.12], [784, 0.2, 0.22]]),
    drop: () => play("sine", 0.14, [[660, 0, 0.14], [523, 0.1, 0.14], [392, 0.2, 0.3]]),
    grab: () => play("sine", 0.16, [[330, 0, 0.09], [440, 0.09, 0.16]]),
    win: () => play("triangle", 0.2, [[523, 0, 0.16], [659, 0.12, 0.16], [784, 0.24, 0.16], [1047, 0.36, 0.5], [1319, 0.5, 0.7]]),
    miss: () => play("sine", 0.14, [[392, 0, 0.24], [330, 0.2, 0.24], [262, 0.4, 0.55]]),
    motor: (active) => {
      if (!context || !motor) return;
      const now = context.currentTime;
      motor.gain.gain.setTargetAtTime(active ? 0.07 : 0, now, active ? 0.08 : 0.14);
      motor.voices.forEach((voice, index) => {
        const pitch = MOTOR_PITCH[index] ?? 0;
        voice.frequency.setTargetAtTime(active ? pitch : pitch * 0.8, now, 0.2);
      });
    },
    dispose: () => {
      void context?.close();
    },
  };
}
