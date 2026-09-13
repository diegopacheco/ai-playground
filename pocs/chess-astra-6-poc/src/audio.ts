const beat = 0.48;
const melody = [76, 79, 83, 81, 78, 74, 79, 83, 86, 84, 83, 79, 81, 77, 74, 76, 79, 83, 78, 75, 71, 76, 78, 83, 88, 86, 83, 84, 81, 77, 83, 79, 76, 81, 78, 74, 79, 83, 86, 84, 81, 77, 78, 75, 71, 76, 83, 88];
const chords = [[52, 55, 59], [50, 54, 57], [48, 52, 55], [55, 59, 62], [53, 57, 60], [48, 52, 55], [47, 51, 54], [52, 55, 59]];

async function score(sampleRate: number): Promise<AudioBuffer> {
  const duration = melody.length * beat;
  const offline = new OfflineAudioContext(2, Math.ceil((duration + 2) * sampleRate), sampleRate);
  const reverb = offline.createConvolver();
  const impulse = offline.createBuffer(2, sampleRate * 2, sampleRate);
  for (let channel = 0; channel < 2; channel++) {
    const data = impulse.getChannelData(channel);
    for (let i = 0; i < data.length; i++) data[i] = Math.sin(i * 78.233 + channel * 23.17) * Math.pow(1 - i / data.length, 3) * 0.2;
  }
  reverb.buffer = impulse;
  const wet = offline.createGain();
  wet.gain.value = 0.22;
  reverb.connect(wet).connect(offline.destination);
  const note = (midi: number, time: number, length: number, volume: number, type: OscillatorType, pan: number, partial = 1) => {
    const oscillator = offline.createOscillator();
    const envelope = offline.createGain();
    const stereo = offline.createStereoPanner();
    oscillator.type = type;
    oscillator.frequency.value = 440 * 2 ** ((midi - 69) / 12) * partial;
    stereo.pan.value = pan;
    envelope.gain.setValueAtTime(0, time);
    envelope.gain.linearRampToValueAtTime(volume, time + (type === 'triangle' ? 0.18 : 0.008));
    envelope.gain.exponentialRampToValueAtTime(0.0001, time + length);
    oscillator.connect(envelope).connect(stereo);
    stereo.connect(offline.destination);
    stereo.connect(reverb);
    oscillator.start(time);
    oscillator.stop(time + length);
  };
  melody.forEach((midi, i) => {
    const time = i * beat;
    note(midi, time, 1.7, 0.12, 'sine', -0.15);
    note(midi, time, 0.9, 0.035, 'sine', 0.2, 2);
    note(midi, time, 0.4, 0.012, 'sine', 0, 3.01);
    if (i % 3 === 0) {
      const chord = chords[Math.floor(i / 3) % chords.length];
      chord.forEach((tone, index) => {
        note(tone, time, beat * 3, 0.019, 'triangle', index / 2 - 0.5);
        note(tone + 12, time + index * beat, 0.7, 0.04, 'sine', 0.4);
      });
      note(chord[0] - 12, time, 1.2, 0.065, 'sine', 0);
    }
  });
  const rendered = await offline.startRendering();
  const length = Math.round(duration * sampleRate);
  const loop = new AudioBuffer({ length, numberOfChannels: 2, sampleRate });
  for (let channel = 0; channel < 2; channel++) {
    const source = rendered.getChannelData(channel);
    const target = loop.getChannelData(channel);
    target.set(source.subarray(0, length));
    for (let i = length; i < source.length; i++) target[i - length] += source[i];
  }
  return loop;
}

async function theme(context: AudioContext) {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}opening-theme.mp3`);
    if (response.ok && response.headers.get('content-type')?.startsWith('audio/')) return { buffer: await context.decodeAudioData(await response.arrayBuffer()), title: 'OPENING THEME' };
  } catch {}
  return { buffer: await score(context.sampleRate), title: 'THE LIBRARY WALTZ' };
}

export class GameAudio {
  title = 'THE LIBRARY WALTZ';
  private context: AudioContext | null = null;
  private master: GainNode | null = null;
  private music: AudioBufferSourceNode | null = null;
  private track: Promise<{ buffer: AudioBuffer; title: string }> | null = null;
  private hiss: AudioBuffer | null = null;
  private enabled = false;
  private generation = 0;

  async setEnabled(enabled: boolean) {
    this.enabled = enabled;
    const generation = ++this.generation;
    this.music?.stop();
    this.music?.disconnect();
    this.music = null;
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
    this.track ??= theme(this.context);
    const { buffer, title } = await this.track;
    this.title = title;
    if (!this.enabled || generation !== this.generation) return;
    this.music = this.context.createBufferSource();
    this.music.buffer = buffer;
    this.music.loop = true;
    this.music.connect(this.master);
    this.music.start();
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
