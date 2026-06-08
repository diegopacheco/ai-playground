const Ring = (() => {
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

  function bell() {
    ensure();
    const t = actx.currentTime;
    const partials = [784, 1174, 1568, 2350];
    partials.forEach((f, i) => {
      const o = actx.createOscillator(), g = actx.createGain();
      o.type = "sine";
      o.frequency.value = f * (1 + i * 0.002);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.22 / (i + 1), t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 1.1 - i * 0.15);
      o.connect(g); g.connect(master);
      o.start(t); o.stop(t + 1.2);
    });
  }

  function whoosh() {
    ensure();
    const t = actx.currentTime;
    const s = actx.createBufferSource(); s.buffer = nbuf;
    const bp = actx.createBiquadFilter(); bp.type = "bandpass"; bp.Q.value = 1.0;
    bp.frequency.setValueAtTime(420, t);
    bp.frequency.exponentialRampToValueAtTime(2600, t + 0.16);
    const g = actx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.6, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
    s.connect(bp); bp.connect(g); g.connect(master);
    s.start(t); s.stop(t + 0.22);
    const o = actx.createOscillator(), og = actx.createGain();
    o.type = "triangle";
    o.frequency.setValueAtTime(360, t);
    o.frequency.exponentialRampToValueAtTime(120, t + 0.14);
    og.gain.setValueAtTime(0.0001, t);
    og.gain.exponentialRampToValueAtTime(0.3, t + 0.02);
    og.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
    o.connect(og); og.connect(master);
    o.start(t); o.stop(t + 0.18);
  }

  function thud() {
    ensure();
    const t = actx.currentTime;
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(180, t);
    o.frequency.exponentialRampToValueAtTime(60, t + 0.14);
    g.gain.setValueAtTime(0.8, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + 0.24);
    const s = actx.createBufferSource(); s.buffer = nbuf;
    const lp = actx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 900;
    const ng = actx.createGain();
    ng.gain.setValueAtTime(0.5, t);
    ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.1);
    s.connect(lp); lp.connect(ng); ng.connect(master);
    s.start(t); s.stop(t + 0.12);
  }

  function clink() {
    ensure();
    const t = actx.currentTime;
    [2100, 3300].forEach((f) => {
      const o = actx.createOscillator(), g = actx.createGain();
      o.type = "square";
      o.frequency.value = f;
      g.gain.setValueAtTime(0.16, t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
      o.connect(g); g.connect(master);
      o.start(t); o.stop(t + 0.14);
    });
  }

  function crowd() {
    ensure();
    const t = actx.currentTime;
    const s = actx.createBufferSource(); s.buffer = nbuf; s.loop = true;
    const bp = actx.createBiquadFilter(); bp.type = "bandpass"; bp.Q.value = 1.1; bp.frequency.value = 900;
    const g = actx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.linearRampToValueAtTime(0.4, t + 0.3);
    g.gain.linearRampToValueAtTime(0.0001, t + 1.6);
    s.connect(bp); bp.connect(g); g.connect(master);
    s.start(t); s.stop(t + 1.7);
  }

  return { ensure, bell, whoosh, thud, clink, crowd };
})();

window.Ring = Ring;
