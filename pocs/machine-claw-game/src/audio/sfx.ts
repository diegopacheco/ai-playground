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

export function createSfx(): Sfx {
  let context: AudioContext | undefined;
  let hum: GainNode | undefined;

  function unlock(): void {
    if (context) {
      void context.resume();
      return;
    }
    context = new AudioContext();
    const oscillator = context.createOscillator();
    hum = context.createGain();
    oscillator.type = "sawtooth";
    oscillator.frequency.value = 62;
    hum.gain.value = 0;
    oscillator.connect(hum).connect(context.destination);
    oscillator.start();
  }

  function play(type: OscillatorType, volume: number, notes: readonly Note[]): void {
    if (!context) return;
    for (const [frequency, delay, length] of notes) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const at = context.currentTime + delay;
      oscillator.type = type;
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(volume, at);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + length);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(at);
      oscillator.stop(at + length);
    }
  }

  return {
    unlock,
    coin: () => play("triangle", 0.25, [[1320, 0, 0.09], [1980, 0.07, 0.35]]),
    start: () => play("square", 0.08, [[523, 0, 0.1], [659, 0.1, 0.1], [784, 0.2, 0.18]]),
    drop: () => play("sawtooth", 0.1, [[440, 0, 0.12], [330, 0.1, 0.12], [220, 0.2, 0.25]]),
    grab: () => play("square", 0.09, [[180, 0, 0.07], [140, 0.08, 0.1]]),
    win: () => play("triangle", 0.22, [[523, 0, 0.14], [659, 0.12, 0.14], [784, 0.24, 0.14], [1047, 0.36, 0.5], [1319, 0.5, 0.6]]),
    miss: () => play("sawtooth", 0.09, [[294, 0, 0.22], [247, 0.2, 0.22], [196, 0.4, 0.5]]),
    motor: (active) => {
      if (context && hum) hum.gain.setTargetAtTime(active ? 0.035 : 0, context.currentTime, 0.05);
    },
    dispose: () => {
      void context?.close();
    },
  };
}
