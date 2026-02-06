var channel = new BroadcastChannel('tetris-sync');

var DEFAULT_KEYMAP = {
    left: ['ArrowLeft'],
    right: ['ArrowRight'],
    down: ['ArrowDown'],
    rotate: ['ArrowUp'],
    hardDrop: ['Space'],
    hold: ['KeyC', 'ShiftLeft', 'ShiftRight'],
    pause: ['KeyP']
};

var ACTION_LABELS = {
    left: 'Move Left',
    right: 'Move Right',
    down: 'Soft Drop',
    rotate: 'Rotate',
    hardDrop: 'Hard Drop',
    hold: 'Hold Piece',
    pause: 'Pause'
};

var KEY_DISPLAY_NAMES = {
    ArrowLeft: 'Left Arrow',
    ArrowRight: 'Right Arrow',
    ArrowUp: 'Up Arrow',
    ArrowDown: 'Down Arrow',
    Space: 'Space',
    ShiftLeft: 'Left Shift',
    ShiftRight: 'Right Shift',
    KeyC: 'C',
    KeyP: 'P',
    KeyA: 'A',
    KeyD: 'D',
    KeyS: 'S',
    KeyW: 'W',
    KeyQ: 'Q',
    KeyE: 'E',
    KeyZ: 'Z',
    KeyX: 'X',
    Enter: 'Enter',
    Escape: 'Escape',
    Tab: 'Tab',
    Backspace: 'Backspace',
    ControlLeft: 'Left Ctrl',
    ControlRight: 'Right Ctrl',
    AltLeft: 'Left Alt',
    AltRight: 'Right Alt'
};

var capturingAction = null;
var currentKeymap = {};

function getKeyDisplayName(code) {
    if (KEY_DISPLAY_NAMES[code]) {
        return KEY_DISPLAY_NAMES[code];
    }
    if (code.indexOf('Key') === 0) {
        return code.charAt(3);
    }
    if (code.indexOf('Digit') === 0) {
        return code.charAt(5);
    }
    return code;
}

function getDefaultKeymap() {
    var copy = {};
    for (var action in DEFAULT_KEYMAP) {
        copy[action] = DEFAULT_KEYMAP[action].slice();
    }
    return copy;
}

function initKeyBindingUI() {
    try {
        var stored = localStorage.getItem('tetris_keybindings');
        if (stored) {
            currentKeymap = JSON.parse(stored);
        } else {
            currentKeymap = getDefaultKeymap();
        }
    } catch (e) {
        currentKeymap = getDefaultKeymap();
    }
    updateAllBindingButtons();
    for (var action in ACTION_LABELS) {
        setupBindButton(action);
    }
    document.getElementById('restore-defaults').addEventListener('click', function() {
        currentKeymap = getDefaultKeymap();
        saveAndSync();
        updateAllBindingButtons();
    });
}

function setupBindButton(action) {
    var button = document.getElementById('bind-' + action);
    if (button) {
        button.addEventListener('click', function() {
            startCapture(action, button);
        });
    }
}

function startCapture(action, button) {
    if (capturingAction !== null) {
        cancelCapture();
    }
    capturingAction = action;
    button.textContent = 'Press a key...';
    button.classList.add('capturing');
    document.addEventListener('keydown', handleKeyCapture);
}

function handleKeyCapture(event) {
    event.preventDefault();
    event.stopPropagation();
    if (capturingAction === null) {
        return;
    }
    var code = event.code;
    var conflict = findConflict(code, capturingAction);
    if (conflict) {
        alert('Key "' + getKeyDisplayName(code) + '" is already bound to ' + ACTION_LABELS[conflict]);
        cancelCapture();
        return;
    }
    currentKeymap[capturingAction] = [code];
    saveAndSync();
    var button = document.getElementById('bind-' + capturingAction);
    if (button) {
        button.textContent = getKeyDisplayName(code);
        button.classList.remove('capturing');
    }
    document.removeEventListener('keydown', handleKeyCapture);
    capturingAction = null;
}

function cancelCapture() {
    if (capturingAction === null) {
        return;
    }
    var button = document.getElementById('bind-' + capturingAction);
    if (button && currentKeymap[capturingAction] && currentKeymap[capturingAction].length > 0) {
        var keys = currentKeymap[capturingAction];
        var names = [];
        for (var i = 0; i < keys.length; i++) {
            names.push(getKeyDisplayName(keys[i]));
        }
        button.textContent = names.join(', ');
    }
    if (button) {
        button.classList.remove('capturing');
    }
    document.removeEventListener('keydown', handleKeyCapture);
    capturingAction = null;
}

function findConflict(code, excludeAction) {
    for (var action in currentKeymap) {
        if (action === excludeAction) {
            continue;
        }
        if (currentKeymap[action] && currentKeymap[action].indexOf(code) !== -1) {
            return action;
        }
    }
    return null;
}

function saveAndSync() {
    localStorage.setItem('tetris_keybindings', JSON.stringify(currentKeymap));
    channel.postMessage({ type: 'KEYMAP_CHANGE', keymap: currentKeymap });
}

function updateAllBindingButtons() {
    for (var action in currentKeymap) {
        var button = document.getElementById('bind-' + action);
        if (button && currentKeymap[action] && currentKeymap[action].length > 0) {
            var keys = currentKeymap[action];
            var names = [];
            for (var i = 0; i < keys.length; i++) {
                names.push(getKeyDisplayName(keys[i]));
            }
            button.textContent = names.join(', ');
        }
    }
}

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
    else if (data.type === 'KEYMAP_CHANGE') {
        if (data.keymap) {
            currentKeymap = data.keymap;
            updateAllBindingButtons();
        }
    }
};

window.addEventListener('beforeunload', function() {
    channel.close();
});

initKeyBindingUI();
