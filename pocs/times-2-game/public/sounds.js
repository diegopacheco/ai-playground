const Sounds = (() => {
  let ctx = null;
  let enabled = true;

  function ensure() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  function tone(freq, start, duration, type, gainPeak) {
    const ac = ensure();
    const osc = ac.createOscillator();
    const gain = ac.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, ac.currentTime + start);
    gain.gain.setValueAtTime(0, ac.currentTime + start);
    gain.gain.linearRampToValueAtTime(gainPeak, ac.currentTime + start + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + start + duration);
    osc.connect(gain).connect(ac.destination);
    osc.start(ac.currentTime + start);
    osc.stop(ac.currentTime + start + duration + 0.02);
  }

  function play(name) {
    if (!enabled) return;
    try {
      if (name === "train") {
        tone(523, 0, 0.09, "triangle", 0.18);
        tone(784, 0.08, 0.1, "triangle", 0.18);
      } else if (name === "coin") {
        tone(880, 0, 0.05, "square", 0.1);
        tone(1175, 0.04, 0.06, "square", 0.1);
      } else if (name === "hit") {
        tone(160, 0, 0.08, "sawtooth", 0.16);
      } else if (name === "attack") {
        tone(330, 0, 0.12, "sawtooth", 0.2);
        tone(247, 0.1, 0.18, "square", 0.18);
      } else if (name === "destroy") {
        tone(120, 0, 0.3, "sawtooth", 0.25);
        tone(80, 0.12, 0.35, "triangle", 0.2);
      } else if (name === "victory") {
        tone(523, 0, 0.14, "triangle", 0.22);
        tone(659, 0.13, 0.14, "triangle", 0.22);
        tone(784, 0.26, 0.14, "triangle", 0.22);
        tone(1046, 0.39, 0.26, "triangle", 0.24);
      } else if (name === "defeat") {
        tone(392, 0, 0.2, "sawtooth", 0.2);
        tone(294, 0.18, 0.22, "sawtooth", 0.2);
        tone(196, 0.36, 0.4, "sawtooth", 0.2);
      }
    } catch (e) {}
  }

  function setEnabled(v) { enabled = v; }

  return { play, setEnabled };
})();
window.Sounds = Sounds;
