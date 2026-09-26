export function createSound() {
  let ctx = null;
  let master = null;
  let volume = 0.7;
  let on = false;

  function noiseBuffer(seconds, brown) {
    const buf = ctx.createBuffer(1, ctx.sampleRate * seconds, ctx.sampleRate);
    const data = buf.getChannelData(0);
    let last = 0;
    for (let i = 0; i < data.length; i++) {
      const white = Math.random() * 2 - 1;
      last = brown ? (last + 0.02 * white) / 1.02 : white;
      data[i] = brown ? last * 3.5 : white;
    }
    return buf;
  }

  function ambience() {
    const src = ctx.createBufferSource();
    src.buffer = noiseBuffer(6, true);
    src.loop = true;
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 420;
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.11;
    const depth = ctx.createGain();
    depth.gain.value = 180;
    lfo.connect(depth).connect(filter.frequency);
    const gain = ctx.createGain();
    gain.gain.value = 0.35;
    src.connect(filter).connect(gain).connect(master);
    const hum = ctx.createOscillator();
    hum.type = 'triangle';
    hum.frequency.value = 58;
    const humGain = ctx.createGain();
    humGain.gain.value = 0.012;
    hum.connect(humGain).connect(master);
    src.start();
    lfo.start();
    hum.start();
  }

  function envelope(peak, attack, release, at = ctx.currentTime) {
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(peak, at + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, at + attack + release);
    g.connect(master);
    return g;
  }

  function sweep(type, from, to, peak, duration, at = ctx.currentTime) {
    const osc = ctx.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(from, at);
    osc.frequency.exponentialRampToValueAtTime(to, at + duration);
    osc.connect(envelope(peak, 0.008, duration, at));
    osc.start(at);
    osc.stop(at + duration + 0.05);
  }

  function burst(freq, peak, duration) {
    const src = ctx.createBufferSource();
    src.buffer = noiseBuffer(duration + 0.05, false);
    const band = ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.frequency.value = freq;
    band.Q.value = 1.4;
    src.connect(band).connect(envelope(peak, 0.005, duration));
    src.start();
  }

  const ready = () => on && ctx && ctx.state === 'running';

  return {
    async enable() {
      if (!ctx) {
        ctx = new AudioContext();
        master = ctx.createGain();
        master.gain.value = volume;
        master.connect(ctx.destination);
        ambience();
      }
      on = true;
      await ctx.resume();
    },
    async disable() {
      on = false;
      if (ctx) await ctx.suspend();
    },
    setVolume(v) {
      volume = v;
      if (master) master.gain.setTargetAtTime(v, ctx.currentTime, 0.05);
    },
    bubble() {
      if (!ready()) return;
      const f = 320 + Math.random() * 520;
      sweep('sine', f, f * 2.4, 0.09, 0.07);
    },
    splash() {
      if (!ready()) return;
      burst(1400, 0.35, 0.18);
      sweep('sine', 900, 160, 0.25, 0.16);
    },
    scoop() {
      if (!ready()) return;
      sweep('sine', 180, 760, 0.2, 0.14);
    },
    thunk() {
      if (!ready()) return;
      sweep('sine', 130, 55, 0.45, 0.3);
      burst(300, 0.2, 0.2);
    },
    munch() {
      if (!ready()) return;
      sweep('square', 520, 260, 0.04, 0.05);
    },
    click() {
      if (!ready()) return;
      sweep('triangle', 1500, 1100, 0.05, 0.04);
    },
    jaws() {
      if (!ready()) return;
      const t0 = ctx.currentTime;
      let at = t0;
      let gap = 0.55;
      for (let i = 0; i < 12; i++) {
        const osc = ctx.createOscillator();
        osc.type = 'sawtooth';
        osc.frequency.value = i % 2 ? 87.31 : 82.41;
        const low = ctx.createBiquadFilter();
        low.type = 'lowpass';
        low.frequency.value = 380;
        osc.connect(low).connect(envelope(0.22 + i * 0.015, 0.02, gap * 0.8, at));
        osc.start(at);
        osc.stop(at + gap);
        at += gap;
        gap = Math.max(0.16, gap * 0.84);
      }
    }
  };
}
