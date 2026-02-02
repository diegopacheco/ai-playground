const keys = {};
const keyTimers = {};
const DAS_DELAY = 170;
const DAS_REPEAT = 50;

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
}

function getInput() {
    const now = Date.now();
    const input = {
        left: false,
        right: false,
        down: false,
        rotate: false,
        hardDrop: false
    };

    if (keys['ArrowLeft']) {
        const timer = keyTimers['ArrowLeft'];
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

    if (keys['ArrowRight']) {
        const timer = keyTimers['ArrowRight'];
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

    if (keys['ArrowDown']) {
        input.down = true;
    }

    if (keys['ArrowUp']) {
        const timer = keyTimers['ArrowUp'];
        if (timer && timer.lastRepeat === 0) {
            input.rotate = true;
            timer.lastRepeat = now;
        }
    }

    if (keys['Space']) {
        const timer = keyTimers['Space'];
        if (timer && timer.lastRepeat === 0) {
            input.hardDrop = true;
            timer.lastRepeat = now;
        }
    }

    return input;
}
