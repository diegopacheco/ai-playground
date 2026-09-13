export class GameAudio {
  private context: AudioContext | null = null;
  private master: GainNode | null = null;
  private hiss: AudioBuffer | null = null;
  private enabled = false;

  async setEnabled(enabled: boolean) {
    this.enabled = enabled;
    if (!enabled) {
      if (this.master && this.context) this.master.gain.setTargetAtTime(0, this.context.currentTime, 0.025);
      return;
    }
    this.context ??= new AudioContext();
    this.master ??= this.context.createGain();
    this.master.disconnect();
    this.master.connect(this.context.destination);
    this.master.gain.value = 0.65;
    await this.context.resume();
  }

  move() {
    [523, 784].forEach((frequency, i) => this.tone(frequency, frequency, i * 0.06, 0.65 - i * 0.06, 0.08));
  }

  capture() {
    this.noise(0, 0.6, 0.12, 3000, 400);
    this.tone(880, 140, 0, 0.8, 0.07, 'triangle');
    [1568, 1319, 1047].forEach((frequency, i) => this.tone(frequency, frequency, 0.25 + i * 0.08, 0.3, 0.03));
  }

  checkmate() {
    [55, 82.4, 116.5].forEach(frequency => this.tone(frequency, frequency * 0.97, 0, 2.8, 0.09, 'triangle'));
    this.noise(0, 0.9, 0.06, 200, 900);
    this.tone(120, 40, 0.9, 0.5, 0.3);
    this.noise(0.9, 0.9, 0.18, 4000, 300);
    [1319, 988, 784, 659, 494].forEach((frequency, i) => this.tone(frequency, frequency, 1 + i * 0.12, 1.2, 0.04));
  }

  private tone(frequency: number, end: number, delay: number, length: number, volume: number, type: OscillatorType = 'sine') {
    if (!this.enabled || !this.context || !this.master) return;
    const at = this.context.currentTime + delay;
    const oscillator = this.context.createOscillator();
    const envelope = this.context.createGain();
    oscillator.type = type;
    oscillator.frequency.setValueAtTime(frequency, at);
    oscillator.frequency.exponentialRampToValueAtTime(end, at + length);
    envelope.gain.setValueAtTime(0, at);
    envelope.gain.linearRampToValueAtTime(volume, at + 0.02);
    envelope.gain.exponentialRampToValueAtTime(0.0001, at + length);
    oscillator.connect(envelope).connect(this.master);
    oscillator.onended = () => { oscillator.disconnect(); envelope.disconnect(); };
    oscillator.start(at);
    oscillator.stop(at + length + 0.05);
  }

  private noise(delay: number, length: number, volume: number, from: number, to: number) {
    if (!this.enabled || !this.context || !this.master) return;
    const at = this.context.currentTime + delay;
    if (!this.hiss) {
      this.hiss = this.context.createBuffer(1, this.context.sampleRate, this.context.sampleRate);
      const data = this.hiss.getChannelData(0);
      for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    }
    const source = this.context.createBufferSource();
    const filter = this.context.createBiquadFilter();
    const envelope = this.context.createGain();
    source.buffer = this.hiss;
    filter.type = 'bandpass';
    filter.Q.value = 0.8;
    filter.frequency.setValueAtTime(from, at);
    filter.frequency.exponentialRampToValueAtTime(to, at + length);
    envelope.gain.setValueAtTime(0, at);
    envelope.gain.linearRampToValueAtTime(volume, at + 0.03);
    envelope.gain.exponentialRampToValueAtTime(0.0001, at + length);
    source.connect(filter).connect(envelope).connect(this.master);
    source.onended = () => { source.disconnect(); filter.disconnect(); envelope.disconnect(); };
    source.start(at);
    source.stop(at + length);
  }
}
