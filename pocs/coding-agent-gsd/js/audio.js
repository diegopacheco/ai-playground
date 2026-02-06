let audioContext = null;
let muted = false;
let channel = null;

function getAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
    return audioContext;
}

function initAudio() {
    const storedMute = localStorage.getItem('audio_muted');
    if (storedMute !== null) {
        muted = storedMute === 'true';
    }

    channel = new BroadcastChannel('tetris-sync');
    channel.addEventListener('message', (event) => {
        if (event.data.type === 'MUTE_CHANGE') {
            muted = event.data.muted;
        }
    });

    document.addEventListener('click', () => {
        const ctx = getAudioContext();
        if (ctx.state === 'suspended') {
            ctx.resume();
        }
    }, { once: true });
}

function isMuted() {
    return muted;
}

function setMuted(value) {
    muted = value;
    localStorage.setItem('audio_muted', String(value));
    if (channel) {
        channel.postMessage({
            type: 'MUTE_CHANGE',
            muted: value
        });
    }
}

function playSound(frequency, duration) {
    if (muted) return;

    const ctx = getAudioContext();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();

    oscillator.connect(gain);
    gain.connect(ctx.destination);

    const nyquistLimit = ctx.sampleRate / 2 * 0.9;
    const cappedFrequency = Math.min(frequency, nyquistLimit);

    oscillator.frequency.setValueAtTime(cappedFrequency, ctx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(cappedFrequency, ctx.currentTime + 0.03);
    oscillator.type = 'sine';

    gain.gain.value = 0.1;
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration / 1000);

    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + duration / 1000);
}

function playLandSound() {
    playSound(220, 100);
}

function playLineClearSound(linesCleared, combo) {
    const baseFrequencies = {
        1: 330,
        2: 440,
        3: 550,
        4: 660
    };
    const baseFrequency = baseFrequencies[linesCleared] || 440;
    const scaledFrequency = baseFrequency * (1 + 0.1 * Math.min(combo || 0, 10));
    playSound(scaledFrequency, 100);
}

function playTetrisSound(combo) {
    const baseFrequency = 660;
    const scaledFrequency = baseFrequency * (1 + 0.1 * Math.min(combo || 0, 10));
    playSound(scaledFrequency, 200);
}

function playGameOverSound() {
    playSound(110, 500);
}
