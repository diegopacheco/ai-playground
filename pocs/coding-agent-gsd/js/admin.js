var channel = new BroadcastChannel('tetris-sync');

var muteToggle = document.getElementById('mute-toggle');
var storedMute = localStorage.getItem('audio_muted');
if (storedMute !== null) {
    muteToggle.checked = storedMute === 'true';
}
muteToggle.addEventListener('change', function() {
    var muted = muteToggle.checked;
    localStorage.setItem('audio_muted', String(muted));
    channel.postMessage({
        type: 'MUTE_CHANGE',
        payload: { muted: muted }
    });
});

var themeRadios = document.querySelectorAll('input[name="theme"]');
themeRadios.forEach(function(radio) {
    radio.addEventListener('change', function() {
        channel.postMessage({
            type: 'THEME_CHANGE',
            payload: { themeName: radio.value }
        });
    });
});

var speedSlider = document.getElementById('speed-slider');
var speedValue = document.getElementById('speed-value');
speedSlider.addEventListener('input', function() {
    speedValue.textContent = speedSlider.value + 'ms';
    channel.postMessage({
        type: 'SPEED_CHANGE',
        payload: { dropInterval: parseInt(speedSlider.value) }
    });
});

var pointsSlider = document.getElementById('points-slider');
var pointsValue = document.getElementById('points-value');
pointsSlider.addEventListener('input', function() {
    pointsValue.textContent = pointsSlider.value + ' pts';
    channel.postMessage({
        type: 'POINTS_CHANGE',
        payload: { pointsPerRow: parseInt(pointsSlider.value) }
    });
});

var growthSlider = document.getElementById('growth-slider');
var growthValue = document.getElementById('growth-value');
growthSlider.addEventListener('input', function() {
    var seconds = Math.round(parseInt(growthSlider.value) / 1000);
    growthValue.textContent = seconds + 's';
    channel.postMessage({
        type: 'GROWTH_INTERVAL_CHANGE',
        payload: { interval: parseInt(growthSlider.value) }
    });
});

setInterval(function() {
    channel.postMessage({ type: 'STATS_REQUEST', payload: {} });
}, 1000);

channel.onmessage = function(event) {
    var data = event.data;
    if (data.type === 'STATS_RESPONSE') {
        document.getElementById('stat-score').textContent = data.payload.score;
        document.getElementById('stat-level').textContent = data.payload.level;
        document.getElementById('stat-theme').textContent = data.payload.theme;

        var statusEl = document.getElementById('stat-status');
        if (data.payload.gameOver) {
            statusEl.textContent = 'Game Over';
            statusEl.className = 'stat-value status gameover';
        } else if (data.payload.paused) {
            statusEl.textContent = 'Paused';
            statusEl.className = 'stat-value status paused';
        } else {
            statusEl.textContent = 'Playing';
            statusEl.className = 'stat-value status playing';
        }

        var themeRadio = document.querySelector('input[name="theme"][value="' + data.payload.theme + '"]');
        if (themeRadio) themeRadio.checked = true;
    }
    else if (data.type === 'THEME_CHANGE') {
        var themeRadio = document.querySelector('input[name="theme"][value="' + data.payload.themeName + '"]');
        if (themeRadio) themeRadio.checked = true;
    }
    else if (data.type === 'MUTE_CHANGE') {
        muteToggle.checked = data.payload.muted;
    }
};

window.addEventListener('beforeunload', function() {
    channel.close();
});
