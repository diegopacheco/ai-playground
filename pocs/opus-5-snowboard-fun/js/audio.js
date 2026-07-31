export class Sound {
  constructor() {
    this.ctx = null;
  }

  start() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume();
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    this.ctx = ctx;

    const len = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    this.noise = buf;

    this.master = ctx.createGain();
    this.master.gain.value = 0.85;
    this.master.connect(ctx.destination);

    this.wind = ctx.createBufferSource();
    this.wind.buffer = buf;
    this.wind.loop = true;
    const windFilter = ctx.createBiquadFilter();
    windFilter.type = 'bandpass';
    windFilter.frequency.value = 420;
    windFilter.Q.value = 0.6;
    this.windGain = ctx.createGain();
    this.windGain.gain.value = 0;
    this.wind.connect(windFilter).connect(this.windGain).connect(this.master);
    this.wind.start();

    this.carve = ctx.createBufferSource();
    this.carve.buffer = buf;
    this.carve.loop = true;
    this.carveFilter = ctx.createBiquadFilter();
    this.carveFilter.type = 'highpass';
    this.carveFilter.frequency.value = 2600;
    this.carveGain = ctx.createGain();
    this.carveGain.gain.value = 0;
    this.carve.connect(this.carveFilter).connect(this.carveGain).connect(this.master);
    this.carve.start();
  }

  update(rider) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const speed = rider.speed;
    const wind = Math.min(0.32, Math.max(0, (speed - 4) / 34) ** 2 * 0.4);
    this.windGain.gain.setTargetAtTime(wind, t, 0.12);
    const carve = rider.airborne ? 0 : Math.min(0.3, (0.05 + rider.skid * 0.5) * Math.min(1, speed / 16));
    this.carveGain.gain.setTargetAtTime(carve, t, 0.06);
    this.carveFilter.frequency.setTargetAtTime(1600 + speed * 90 + rider.skid * 1800, t, 0.1);
  }

  burst(kind) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const f = ctx.createBiquadFilter();
    const g = ctx.createGain();
    if (kind === 'crash') {
      f.type = 'lowpass';
      f.frequency.value = 900;
      g.gain.setValueAtTime(0.5, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.9);
      src.stop(t + 0.9);
    } else if (kind === 'land') {
      f.type = 'lowpass';
      f.frequency.value = 520;
      g.gain.setValueAtTime(0.4, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
      src.stop(t + 0.3);
    } else {
      f.type = 'highpass';
      f.frequency.value = 1800;
      g.gain.setValueAtTime(0.18, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
      src.stop(t + 0.25);
    }
    src.connect(f).connect(g).connect(this.master);
    src.start(t);
  }

  quiet() {
    if (!this.ctx) return;
    this.windGain.gain.cancelScheduledValues(this.ctx.currentTime);
    this.carveGain.gain.cancelScheduledValues(this.ctx.currentTime);
    this.windGain.gain.value = 0;
    this.carveGain.gain.value = 0;
    if (this.ctx.state === 'running') this.ctx.suspend();
  }
}
