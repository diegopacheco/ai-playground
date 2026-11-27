export function createSoundscape() {
  let ctx = null;
  let windGain = null;
  let enabled = false;

  function init() {
    if (ctx) return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    ctx = new AudioCtx();

    const bufferSize = ctx.sampleRate * 2;
    const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i += 1) {
      data[i] = Math.random() * 2 - 1;
    }

    const windSource = ctx.createBufferSource();
    windSource.buffer = noiseBuffer;
    windSource.loop = true;

    const windFilter = ctx.createBiquadFilter();
    windFilter.type = 'bandpass';
    windFilter.frequency.value = 380;
    windFilter.Q.value = 1.4;

    windGain = ctx.createGain();
    windGain.gain.value = 0.0;

    windSource.connect(windFilter);
    windFilter.connect(windGain);
    windGain.connect(ctx.destination);
    windSource.start();
  }

  function setEnabled(next) {
    enabled = Boolean(next);
    if (enabled) {
      init();
      if (ctx && ctx.state === 'suspended') {
        ctx.resume();
      }
    }
    if (windGain && ctx) {
      windGain.gain.setTargetAtTime(enabled ? 0.045 : 0.0, ctx.currentTime, 0.2);
    }
    return enabled;
  }

  function triggerHoofbeat(intensity = 1, legIndex = 0) {
    if (!enabled || !ctx || ctx.state !== 'running') return;
    const now = ctx.currentTime;

    const osc = ctx.createOscillator();
    const oscGain = ctx.createGain();
    osc.type = 'triangle';
    const baseFreq = 82 + (legIndex % 2) * 14 + Math.random() * 12;
    osc.frequency.setValueAtTime(baseFreq, now);
    osc.frequency.exponentialRampToValueAtTime(36, now + 0.085);

    oscGain.gain.setValueAtTime(0.18 * intensity, now);
    oscGain.gain.exponentialRampToValueAtTime(0.001, now + 0.095);
    osc.connect(oscGain);
    oscGain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.1);

    const len = Math.floor(ctx.sampleRate * 0.06);
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < len; i += 1) {
      ch[i] = (Math.random() * 2 - 1) * (1 - i / len);
    }
    const crunch = ctx.createBufferSource();
    crunch.buffer = buf;

    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 1250 + Math.random() * 450;
    filter.Q.value = 0.9;

    const cGain = ctx.createGain();
    cGain.gain.setValueAtTime(0.09 * intensity, now);
    cGain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);

    crunch.connect(filter);
    filter.connect(cGain);
    cGain.connect(ctx.destination);
    crunch.start(now);
  }

  return {
    isEnabled: () => enabled,
    setEnabled,
    triggerHoofbeat
  };
}
