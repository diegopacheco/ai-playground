import { SURFACES } from './core/vehicle.js';

function noiseBuffer(ctx, seconds = 2) {
  const buf = ctx.createBuffer(1, ctx.sampleRate * seconds, ctx.sampleRate);
  const data = buf.getChannelData(0);
  let brown = 0;
  for (let i = 0; i < data.length; i++) {
    const white = Math.random() * 2 - 1;
    brown = (brown + 0.02 * white) / 1.02;
    data[i] = white * 0.5 + brown * 3;
  }
  return buf;
}

function loopNoise(ctx, buffer) {
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.loop = true;
  src.start(0, Math.random() * 1.5);
  return src;
}

function distortionCurve(amount) {
  const n = 1024;
  const curve = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1;
    curve[i] = ((3 + amount) * x * 20 * (Math.PI / 180)) / (Math.PI + amount * Math.abs(x));
  }
  return curve;
}

function createEngine(ctx, destination, noise, cylinders, level) {
  const out = ctx.createGain();
  out.gain.value = 0;
  const panner = ctx.createPanner();
  panner.panningModel = 'HRTF';
  panner.distanceModel = 'exponential';
  panner.refDistance = 6;
  panner.rolloffFactor = 1.3;
  out.connect(panner).connect(destination);

  const filter = ctx.createBiquadFilter();
  filter.type = 'lowpass';
  filter.Q.value = 3;
  const shaper = ctx.createWaveShaper();
  shaper.curve = distortionCurve(cylinders === 8 ? 60 : 30);
  shaper.oversample = '2x';
  const mix = ctx.createGain();
  mix.connect(shaper).connect(filter).connect(out);

  const oscs = [
    { type: 'sawtooth', mult: 1, gain: 0.5 },
    { type: 'square', mult: 0.5, gain: 0.35 },
    { type: 'sawtooth', mult: 2, gain: 0.18 },
    { type: 'triangle', mult: 0.25, gain: 0.45 },
  ].map((o) => {
    const osc = ctx.createOscillator();
    osc.type = o.type;
    const g = ctx.createGain();
    g.gain.value = o.gain;
    osc.connect(g).connect(mix);
    osc.start();
    return { osc, mult: o.mult };
  });

  const burble = ctx.createGain();
  burble.gain.value = 0.5;
  const lfo = ctx.createOscillator();
  lfo.type = 'square';
  const lfoDepth = ctx.createGain();
  lfoDepth.gain.value = 0.35;
  lfo.connect(lfoDepth).connect(burble.gain);
  lfo.start();
  mix.connect(burble);

  const intake = loopNoise(ctx, noise);
  const intakeFilter = ctx.createBiquadFilter();
  intakeFilter.type = 'bandpass';
  intakeFilter.Q.value = 1.2;
  const intakeGain = ctx.createGain();
  intake.connect(intakeFilter).connect(intakeGain).connect(out);

  return {
    panner,
    stop() {
      for (const o of oscs) o.osc.stop();
      lfo.stop();
      intake.stop();
      panner.disconnect();
    },
    update(rpm, throttle, t) {
      const fire = (rpm / 60) * (cylinders / 2);
      for (const o of oscs) o.osc.frequency.setTargetAtTime(fire * o.mult, t, 0.03);
      lfo.frequency.setTargetAtTime(fire / (cylinders === 8 ? 4 : 3), t, 0.05);
      filter.frequency.setTargetAtTime(280 + throttle * 2200 + rpm * 0.35, t, 0.05);
      intakeFilter.frequency.setTargetAtTime(400 + rpm * 0.5, t, 0.05);
      intakeGain.gain.setTargetAtTime(0.05 + throttle * 0.22, t, 0.05);
      out.gain.setTargetAtTime(level * (0.2 + throttle * 0.28 + rpm / 26000), t, 0.05);
    },
  };
}

function surfaceLoop(ctx, destination, noise, type, freq, q) {
  const src = loopNoise(ctx, noise);
  const f = ctx.createBiquadFilter();
  f.type = type;
  f.frequency.value = freq;
  f.Q.value = q;
  const g = ctx.createGain();
  g.gain.value = 0;
  src.connect(f).connect(g).connect(destination);
  return { gain: g, filter: f };
}

export function createAudio() {
  const ctx = new AudioContext();
  const master = ctx.createGain();
  master.gain.value = 0.8;
  const comp = ctx.createDynamicsCompressor();
  master.connect(comp).connect(ctx.destination);
  const noise = noiseBuffer(ctx);
  let engines = [];
  let muted = false;
  const tires = surfaceLoop(ctx, master, noise, 'bandpass', 500, 0.8);
  const gravel = surfaceLoop(ctx, master, noise, 'highpass', 2200, 0.5);
  const slide = surfaceLoop(ctx, master, noise, 'bandpass', 900, 4);
  const wind = surfaceLoop(ctx, master, noise, 'lowpass', 600, 0.4);
  const rain = surfaceLoop(ctx, master, noise, 'highpass', 1800, 0.3);

  function blip(freq, dur, vol = 0.3) {
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.value = freq;
    o.type = 'square';
    g.gain.setValueAtTime(vol, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
    o.connect(g).connect(master);
    o.start();
    o.stop(ctx.currentTime + dur);
  }

  return {
    ctx,
    setupRace(cars, playerIndex, weather) {
      engines = cars.map((car, k) => createEngine(ctx, master, noise, car.spec.cylinders, k === playerIndex ? 1 : 0.7));
      rain.gain.gain.value = weather === 'rain' ? 0.12 : 0;
    },
    update(cars, playerIndex, camera) {
      const t = ctx.currentTime;
      const l = ctx.listener;
      const fwd = camera.getWorldDirection(camera.userData.tmp);
      if (l.positionX) {
        l.positionX.setValueAtTime(camera.position.x, t);
        l.positionY.setValueAtTime(camera.position.y, t);
        l.positionZ.setValueAtTime(camera.position.z, t);
        l.forwardX.setValueAtTime(fwd.x, t);
        l.forwardY.setValueAtTime(fwd.y, t);
        l.forwardZ.setValueAtTime(fwd.z, t);
      }
      cars.forEach((car, k) => {
        const e = engines[k];
        if (!e) return;
        const throttle = car.input ? Math.max(car.input.throttle, car.gear === -1 ? car.input.brake : 0) : 0;
        e.update(car.rpm, throttle, t);
        e.panner.positionX.setValueAtTime(car.x, t);
        e.panner.positionY.setValueAtTime(car.y + 0.8, t);
        e.panner.positionZ.setValueAtTime(car.z, t);
      });
      const p = cars[playerIndex];
      const speed = Math.abs(p.u);
      const ground = p.air ? 0 : 1;
      const mud = p.surface === SURFACES.mud || p.surface === SURFACES.puddle;
      tires.filter.frequency.setTargetAtTime(p.surface === SURFACES.puddle ? 260 : mud ? 380 : 700, t, 0.1);
      tires.gain.gain.setTargetAtTime(ground * Math.min(0.35, speed / 70), t, 0.08);
      gravel.gain.gain.setTargetAtTime(ground * (p.surface === SURFACES.offroad ? Math.min(0.2, speed / 110) : speed / 600), t, 0.08);
      slide.gain.gain.setTargetAtTime(ground * Math.min(0.25, p.slip * 0.3 + p.wheelSpin * 0.2) * Math.min(1, speed / 8), t, 0.05);
      wind.gain.gain.setTargetAtTime(Math.min(0.3, (speed * speed) / 9000), t, 0.2);
    },
    impact(strength) {
      const src = ctx.createBufferSource();
      src.buffer = noise;
      const f = ctx.createBiquadFilter();
      f.type = 'lowpass';
      f.frequency.value = 400;
      const g = ctx.createGain();
      g.gain.setValueAtTime(Math.min(1, strength / 10), ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
      src.connect(f).connect(g).connect(master);
      src.start();
      src.stop(ctx.currentTime + 0.4);
    },
    beep(final) {
      blip(final ? 880 : 440, final ? 0.6 : 0.25);
    },
    toggleMute() {
      muted = !muted;
      master.gain.setTargetAtTime(muted ? 0 : 0.8, ctx.currentTime, 0.05);
      return muted;
    },
    stopRace() {
      engines.forEach((e) => e.stop());
      engines = [];
      for (const s of [tires, gravel, slide, wind, rain]) s.gain.gain.value = 0;
    },
  };
}
