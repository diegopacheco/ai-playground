const Sea = (() => {
  let actx = null, master = null, nbuf = null;

  function ensure() {
    if (actx) {
      if (actx.state === "suspended") actx.resume();
      return;
    }
    actx = new (window.AudioContext || window.webkitAudioContext)();
    master = actx.createGain();
    master.gain.value = 0.5;
    master.connect(actx.destination);
    const len = actx.sampleRate;
    nbuf = actx.createBuffer(1, len, actx.sampleRate);
    const d = nbuf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  }

  function plop() {
    ensure();
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(680, t);
    o.frequency.exponentialRampToValueAtTime(180, t + 0.12);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.5, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + 0.18);
  }

  function catch_() {
    ensure();
    const t = actx.currentTime;
    [523, 784, 1047].forEach((f, i) => {
      const o = actx.createOscillator(), g = actx.createGain();
      o.type = "triangle";
      o.frequency.value = f;
      const at = t + i * 0.06;
      g.gain.setValueAtTime(0.0001, at);
      g.gain.exponentialRampToValueAtTime(0.3, at + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 0.22);
      o.connect(g); g.connect(master);
      o.start(at); o.stop(at + 0.24);
    });
  }

  function splash() {
    ensure();
    const t = actx.currentTime;
    const s = actx.createBufferSource(); s.buffer = nbuf;
    const bp = actx.createBiquadFilter(); bp.type = "bandpass"; bp.Q.value = 0.8;
    bp.frequency.setValueAtTime(900, t);
    bp.frequency.exponentialRampToValueAtTime(3200, t + 0.18);
    const g = actx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.5, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.26);
    s.connect(bp); bp.connect(g); g.connect(master);
    s.start(t); s.stop(t + 0.28);
  }

  function reel() {
    ensure();
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "sawtooth";
    o.frequency.setValueAtTime(140, t);
    o.frequency.linearRampToValueAtTime(320, t + 0.18);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.14, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + 0.22);
  }

  function fanfare() {
    ensure();
    const t = actx.currentTime;
    const notes = [523, 659, 784, 1047, 1319];
    notes.forEach((f, i) => {
      const o = actx.createOscillator(), g = actx.createGain();
      o.type = "triangle";
      o.frequency.value = f;
      const at = t + i * 0.12;
      g.gain.setValueAtTime(0.0001, at);
      g.gain.exponentialRampToValueAtTime(0.3, at + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 0.4);
      o.connect(g); g.connect(master);
      o.start(at); o.stop(at + 0.42);
    });
  }

  return { ensure, plop, catch: catch_, splash, reel, fanfare };
})();

window.Sea = Sea;
