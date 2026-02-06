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

    oscillator.frequency.value = frequency;
    oscillator.type = 'sine';

    gain.gain.value = 0.1;
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration / 1000);

    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + duration / 1000);
}

function playLandSound() {
    playSound(220, 100);
}

function playLineClearSound() {
    playSound(440, 100);
}

function playTetrisSound() {
    playSound(880, 200);
}

function playGameOverSound() {
    playSound(110, 500);
}
