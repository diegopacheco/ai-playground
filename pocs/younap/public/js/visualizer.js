const BANDS = 32;
const SAMPLES = 128;
const GAP = 2;

function createSignal(isPlaying) {
  const levels = new Float32Array(BANDS);
  const wave = new Float32Array(SAMPLES);
  const signal = { levels, wave, beat: 0, time: 0 };

  return () => {
    const playing = isPlaying();
    signal.time = performance.now() / 1000;
    signal.beat = playing ? Math.exp(-((signal.time * 2) % 1) * 4) : 0;
    for (let band = 0; band < BANDS; band++) {
      const tilt = 1 - (band / BANDS) * 0.65;
      const kick = band < 6 ? signal.beat * 0.6 : 0;
      const target = playing ? Math.min(1, (Math.random() * 0.65 + kick) * tilt) : 0;
      levels[band] += (target - levels[band]) * 0.3;
    }
    for (let sample = 0; sample < SAMPLES; sample++) {
      const x = sample / SAMPLES;
      const tone = Math.sin(x * 12 + signal.time * 9) * (0.3 + signal.beat * 0.4) + Math.sin(x * 31 - signal.time * 15) * 0.2;
      const target = playing ? tone + (Math.random() - 0.5) * 0.15 : 0;
      wave[sample] += (target - wave[sample]) * 0.5;
    }
    return signal;
  };
}

function traceWave(context, width, height, wave) {
  context.beginPath();
  for (let sample = 0; sample < SAMPLES; sample++) {
    const x = (sample / (SAMPLES - 1)) * width;
    const y = height / 2 + wave[sample] * height * 0.5;
    if (sample) context.lineTo(x, y);
    else context.moveTo(x, y);
  }
  context.stroke();
}

export const MODES = {
  BARS(context, width, height, signal, memo) {
    const barWidth = (width - GAP * (BANDS - 1)) / BANDS;
    context.clearRect(0, 0, width, height);
    for (let band = 0; band < BANDS; band++) {
      const x = band * (barWidth + GAP);
      const barHeight = signal.levels[band] * height;
      memo.peaks[band] = Math.max(signal.levels[band], memo.peaks[band] - 0.012);
      context.fillStyle = memo.gradient;
      context.fillRect(x, height - barHeight, barWidth, barHeight);
      if (memo.peaks[band] > 0.02) {
        context.fillStyle = '#c8c8c8';
        context.fillRect(x, height - memo.peaks[band] * height - 1, barWidth, 1);
      }
    }
  },

  SCOPE(context, width, height, signal) {
    context.clearRect(0, 0, width, height);
    context.strokeStyle = '#00e400';
    context.lineWidth = 2;
    context.shadowColor = '#00e400';
    context.shadowBlur = 8;
    traceWave(context, width, height, signal.wave);
    context.shadowBlur = 0;
  },

  MIRROR(context, width, height, signal) {
    const barWidth = width / (BANDS * 2);
    context.clearRect(0, 0, width, height);
    for (let band = 0; band < BANDS; band++) {
      const half = signal.levels[band] * height * 0.5;
      context.fillStyle = `hsl(${190 + band * 5}, 90%, ${45 + signal.levels[band] * 30}%)`;
      context.fillRect(width / 2 + band * barWidth, height / 2 - half, barWidth - 1, half * 2);
      context.fillRect(width / 2 - (band + 1) * barWidth, height / 2 - half, barWidth - 1, half * 2);
    }
  },

  RADIAL(context, width, height, signal) {
    const radius = height * 0.2 + signal.beat * height * 0.06;
    context.fillStyle = 'rgba(0, 0, 0, 0.25)';
    context.fillRect(0, 0, width, height);
    context.lineWidth = 3;
    for (let spoke = 0; spoke < BANDS * 2; spoke++) {
      const level = signal.levels[spoke < BANDS ? spoke : BANDS * 2 - 1 - spoke];
      const angle = (spoke / (BANDS * 2)) * Math.PI * 2 + signal.time * 0.3;
      const reach = radius + level * height * 0.28;
      context.strokeStyle = `hsl(${(spoke * 6 + signal.time * 40) % 360}, 95%, 55%)`;
      context.beginPath();
      context.moveTo(width / 2 + Math.cos(angle) * radius, height / 2 + Math.sin(angle) * radius);
      context.lineTo(width / 2 + Math.cos(angle) * reach, height / 2 + Math.sin(angle) * reach);
      context.stroke();
    }
  },

  MILK(context, width, height, signal) {
    context.globalAlpha = 0.9;
    context.drawImage(context.canvas, -width * 0.02, -height * 0.03, width * 1.04, height * 1.06);
    context.globalAlpha = 1;
    context.fillStyle = 'rgba(0, 0, 0, 0.06)';
    context.fillRect(0, 0, width, height);
    context.strokeStyle = `hsl(${(signal.time * 60) % 360}, 100%, 60%)`;
    context.lineWidth = 2;
    traceWave(context, width, height, signal.wave);
  }
};

export function createVisualizer(canvas, isPlaying, getMode) {
  const context = canvas.getContext('2d');
  const nextSignal = createSignal(isPlaying);
  const gradient = context.createLinearGradient(0, canvas.height, 0, 0);
  gradient.addColorStop(0, '#00c800');
  gradient.addColorStop(0.6, '#e0e000');
  gradient.addColorStop(1, '#e02000');
  const memo = { gradient, peaks: new Float32Array(BANDS) };

  function step() {
    const draw = MODES[getMode()] || MODES.BARS;
    draw(context, canvas.width, canvas.height, nextSignal(), memo);
    requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}
