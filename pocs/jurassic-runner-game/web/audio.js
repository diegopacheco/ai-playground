const Jungle = (() => {
  let actx = null, master = null, nbuf = null;
  let running = false, muted = false, step = 0, timer = null;

  const PENTA = [196.0, 233.1, 261.6, 311.1, 349.2, 392.0, 466.2];
  const RIFF = [0, 2, 4, 2, 5, 4, 2, 0, -1, 2, 4, 6, 5, 4, 2, 1];

  function ensure() {
    if (actx) return;
    actx = new (window.AudioContext || window.webkitAudioContext)();
    master = actx.createGain();
    master.gain.value = 0.45;
    master.connect(actx.destination);
    const len = actx.sampleRate;
    nbuf = actx.createBuffer(1, len, actx.sampleRate);
    const d = nbuf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  }

  function drone() {
    const f = actx.createBiquadFilter();
    f.type = "lowpass"; f.frequency.value = 380;
    const g = actx.createGain(); g.gain.value = 0.12;
    f.connect(g); g.connect(master);
    for (const fr of [49, 73.5, 98]) {
      const o = actx.createOscillator();
      o.type = "sawtooth"; o.frequency.value = fr;
      o.connect(f); o.start();
    }
    const lfo = actx.createOscillator(), lg = actx.createGain();
    lfo.frequency.value = 0.07; lg.gain.value = 160;
    lfo.connect(lg); lg.connect(f.frequency); lfo.start();
  }

  function kick(t) {
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(140, t);
    o.frequency.exponentialRampToValueAtTime(46, t + 0.16);
    g.gain.setValueAtTime(0.9, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.24);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + 0.26);
  }

  function tom(t, f) {
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "triangle";
    o.frequency.setValueAtTime(f, t);
    o.frequency.exponentialRampToValueAtTime(f * 0.6, t + 0.18);
    g.gain.setValueAtTime(0.55, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + 0.22);
  }

  function shaker(t, v) {
    const s = actx.createBufferSource(); s.buffer = nbuf;
    const hp = actx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 6500;
    const g = actx.createGain();
    g.gain.setValueAtTime(v, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
    s.connect(hp); hp.connect(g); g.connect(master);
    s.start(t); s.stop(t + 0.08);
  }

  function flute(t, semi) {
    const idx = ((semi % PENTA.length) + PENTA.length) % PENTA.length;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "triangle"; o.frequency.value = PENTA[idx] * (semi < 0 ? 0.5 : 1);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.16, t + 0.05);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.42);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + 0.45);
  }

  function tick() {
    const t = actx.currentTime + 0.04;
    const s = step % 16;
    if (s === 0 || s === 6 || s === 8 || s === 11) kick(t);
    if (s === 4 || s === 12) tom(t, 200);
    if (s === 14) tom(t, 150);
    shaker(t, s % 2 ? 0.07 : 0.13);
    if (s % 2 === 0) flute(t, RIFF[s]);
    step++;
  }

  function start() {
    ensure();
    if (actx.state === "suspended") actx.resume();
    if (running) return;
    running = true;
    drone();
    timer = setInterval(tick, 150);
  }

  function roar() {
    ensure();
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    const dist = actx.createWaveShaper();
    const curve = new Float32Array(256);
    for (let i = 0; i < 256; i++) { const x = i / 128 - 1; curve[i] = Math.tanh(x * 3); }
    dist.curve = curve;
    o.type = "sawtooth";
    o.frequency.setValueAtTime(180, t);
    o.frequency.exponentialRampToValueAtTime(58, t + 0.5);
    o.frequency.exponentialRampToValueAtTime(40, t + 1.1);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.6, t + 0.08);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 1.2);
    o.connect(dist); dist.connect(g); g.connect(master);
    o.start(t); o.stop(t + 1.25);
  }

  function jump() {
    ensure();
    if (actx.state === "suspended") actx.resume();
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "triangle";
    o.frequency.setValueAtTime(190, t);
    o.frequency.exponentialRampToValueAtTime(760, t + 0.12);
    o.frequency.exponentialRampToValueAtTime(540, t + 0.24);
    const lfo = actx.createOscillator(), lg = actx.createGain();
    lfo.frequency.value = 24; lg.gain.value = 70;
    lfo.connect(lg); lg.connect(o.frequency); lfo.start(t); lfo.stop(t + 0.32);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.4, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.34);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + 0.36);
  }

  function duck() {
    ensure();
    if (actx.state === "suspended") actx.resume();
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(680, t);
    o.frequency.exponentialRampToValueAtTime(150, t + 0.26);
    const o2 = actx.createOscillator(), g2 = actx.createGain();
    o2.type = "square";
    o2.frequency.setValueAtTime(170, t + 0.24);
    o2.frequency.exponentialRampToValueAtTime(90, t + 0.4);
    g2.gain.setValueAtTime(0.0001, t + 0.24);
    g2.gain.exponentialRampToValueAtTime(0.18, t + 0.28);
    g2.gain.exponentialRampToValueAtTime(0.0001, t + 0.42);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.34, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
    o.connect(g); g.connect(master);
    o2.connect(g2); g2.connect(master);
    o.start(t); o.stop(t + 0.32);
    o2.start(t + 0.24); o2.stop(t + 0.44);
  }

  function toggle() {
    ensure();
    muted = !muted;
    master.gain.setTargetAtTime(muted ? 0 : 0.45, actx.currentTime, 0.05);
    return muted;
  }

  return { start, roar, toggle, jump, duck };
})();

window.Jungle = Jungle;
