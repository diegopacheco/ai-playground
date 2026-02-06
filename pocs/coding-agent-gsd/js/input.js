const keys = {};
const keyTimers = {};
const DAS_DELAY = 170;
const DAS_REPEAT = 50;

const DEFAULT_KEYMAP = {
    left: ['ArrowLeft'],
    right: ['ArrowRight'],
    down: ['ArrowDown'],
    rotate: ['ArrowUp'],
    hardDrop: ['Space'],
    hold: ['KeyC', 'ShiftLeft', 'ShiftRight'],
    pause: ['KeyP']
};

var keymap = {};

var inputChannel = new BroadcastChannel('tetris-sync');

function loadKeymap() {
    try {
        var stored = localStorage.getItem('tetris_keybindings');
        if (stored) {
            keymap = JSON.parse(stored);
        } else {
            restoreDefaults();
        }
    } catch (e) {
        restoreDefaults();
    }
}

function saveKeymap() {
    localStorage.setItem('tetris_keybindings', JSON.stringify(keymap));
    inputChannel.postMessage({ type: 'KEYMAP_CHANGE', keymap: keymap });
}

function restoreDefaults() {
    keymap = {};
    for (var action in DEFAULT_KEYMAP) {
        keymap[action] = DEFAULT_KEYMAP[action].slice();
    }
}

function getKeymap() {
    return keymap;
}

function setKeyBinding(action, keyCode) {
    keymap[action] = [keyCode];
    saveKeymap();
}

inputChannel.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'KEYMAP_CHANGE' && event.data.keymap) {
        keymap = event.data.keymap;
    }
});

function setupInput() {
    document.addEventListener('keydown', function(e) {
        if (!keys[e.code]) {
            keys[e.code] = true;
            keyTimers[e.code] = { pressed: Date.now(), lastRepeat: 0 };
        }
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
            e.preventDefault();
        }
    });

    document.addEventListener('keyup', function(e) {
        keys[e.code] = false;
        delete keyTimers[e.code];
    });

    loadKeymap();
}

function getActiveKeyForAction(action) {
    var boundKeys = keymap[action];
    if (!boundKeys) return null;
    for (var i = 0; i < boundKeys.length; i++) {
        if (keys[boundKeys[i]]) {
            return boundKeys[i];
        }
    }
    return null;
}

function isActionPressed(action) {
    return getActiveKeyForAction(action) !== null;
}

function getInput() {
    const now = Date.now();
    const input = {
        left: false,
        right: false,
        down: false,
        rotate: false,
        hardDrop: false,
        hold: false,
        pause: false
    };

    var leftKey = getActiveKeyForAction('left');
    if (leftKey) {
        const timer = keyTimers[leftKey];
        if (timer) {
            const elapsed = now - timer.pressed;
            if (elapsed < DAS_DELAY) {
                if (timer.lastRepeat === 0) {
                    input.left = true;
                    timer.lastRepeat = now;
                }
            } else {
                if (now - timer.lastRepeat >= DAS_REPEAT) {
                    input.left = true;
                    timer.lastRepeat = now;
                }
            }
        }
    }

    var rightKey = getActiveKeyForAction('right');
    if (rightKey) {
        const timer = keyTimers[rightKey];
        if (timer) {
            const elapsed = now - timer.pressed;
            if (elapsed < DAS_DELAY) {
                if (timer.lastRepeat === 0) {
                    input.right = true;
                    timer.lastRepeat = now;
                }
            } else {
                if (now - timer.lastRepeat >= DAS_REPEAT) {
                    input.right = true;
                    timer.lastRepeat = now;
                }
            }
        }
    }

    if (isActionPressed('down')) {
        input.down = true;
    }

    var rotateKey = getActiveKeyForAction('rotate');
    if (rotateKey) {
        const timer = keyTimers[rotateKey];
        if (timer && timer.lastRepeat === 0) {
            input.rotate = true;
            timer.lastRepeat = now;
        }
    }

    var hardDropKey = getActiveKeyForAction('hardDrop');
    if (hardDropKey) {
        const timer = keyTimers[hardDropKey];
        if (timer && timer.lastRepeat === 0) {
            input.hardDrop = true;
            timer.lastRepeat = now;
        }
    }

    var holdKey = getActiveKeyForAction('hold');
    if (holdKey) {
        const timer = keyTimers[holdKey];
        if (timer && timer.lastRepeat === 0) {
            input.hold = true;
            timer.lastRepeat = now;
        }
    }

    var pauseKey = getActiveKeyForAction('pause');
    if (pauseKey) {
        const timer = keyTimers[pauseKey];
        if (timer && timer.lastRepeat === 0) {
            input.pause = true;
            timer.lastRepeat = now;
        }
    }

    return input;
}
