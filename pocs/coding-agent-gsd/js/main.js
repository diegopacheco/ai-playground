let board;
let currentPiece = null;
let nextPiece = null;
let heldPiece = null;
let canHold = true;
let bag = [];
let lastTime = 0;
let dropCounter = 0;
let dropInterval = 1000;
let lockCounter = 0;
let lockDelay = 500;
let isLocking = false;
let gameOver = false;
let score = 0;
let level = 1;
let clearingLines = [];
let clearingTimer = 0;
let pointsPerRow = 10;
let boardGrowthInterval = 30000;
let themeIndex = 0;
let combo = 0;
let b2bActive = false;
let pendingScoreCalc = null;
let lastAction = null;
let lastKickOffset = null;

const GameState = Object.freeze({
    PLAYING: 'PLAYING',
    FROZEN: 'FROZEN',
    PAUSED: 'PAUSED',
    GAME_OVER: 'GAME_OVER'
});

let gameState = GameState.PLAYING;
let cycleTimer = 0;
const PLAY_DURATION = 10000;
const FREEZE_DURATION = 10000;
let growthTimer = 0;

function calculateLevel() {
    return Math.floor(score / 100) + 1;
}

function checkLevelUp() {
    const newLevel = calculateLevel();
    if (newLevel > level) {
        level = newLevel;
        stats.level = level;
        onLevelUp();
    }
}

function onLevelUp() {
    themeIndex = (themeIndex + 1) % THEME_ORDER.length;
    var newTheme = THEME_ORDER[themeIndex];
    applyTheme(newTheme);
    sendMessage('THEME_CHANGE', { themeName: newTheme });
}

function getGhostY(piece) {
    let ghostY = piece.y;
    while (isValidPosition(board, piece.type, piece.x, ghostY + 1, piece.rotation)) {
        ghostY++;
    }
    return ghostY;
}

function holdPiece() {
    if (!canHold || !currentPiece) return;

    const currentType = currentPiece.type;

    if (heldPiece === null) {
        heldPiece = currentType;
        spawnPiece();
    } else {
        const tempType = heldPiece;
        heldPiece = currentType;

        const piece = PIECES[tempType];
        const shape = piece.shapes[0];
        const startX = Math.floor((COLS - shape[0].length) / 2);

        currentPiece = {
            type: tempType,
            x: startX,
            y: 0,
            rotation: 0
        };
    }

    canHold = false;
    isLocking = false;
    lockCounter = 0;
}

function togglePause() {
    if (gameState === GameState.GAME_OVER) return;
    if (gameState === GameState.PAUSED) {
        gameState = GameState.PLAYING;
    } else if (gameState === GameState.PLAYING || gameState === GameState.FROZEN) {
        gameState = GameState.PAUSED;
    }
}

function shuffleBag() {
    bag = [...PIECE_TYPES];
    for (let i = bag.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [bag[i], bag[j]] = [bag[j], bag[i]];
    }
}

function getNextPiece() {
    if (bag.length === 0) {
        shuffleBag();
    }
    return bag.pop();
}

function spawnPiece() {
    canHold = true;

    if (nextPiece === null) {
        nextPiece = getNextPiece();
    }

    const type = nextPiece;
    nextPiece = getNextPiece();

    const piece = PIECES[type];
    const shape = piece.shapes[0];
    const startX = Math.floor((COLS - shape[0].length) / 2);

    currentPiece = {
        type: type,
        x: startX,
        y: 0,
        rotation: 0
    };

    if (!isValidPosition(board, currentPiece.type, currentPiece.x, currentPiece.y, currentPiece.rotation)) {
        gameOver = true;
        gameState = GameState.GAME_OVER;
        currentPiece = null;
    }

    isLocking = false;
    lockCounter = 0;
}

function movePiece(dx, dy) {
    if (!currentPiece) return false;

    const newX = currentPiece.x + dx;
    const newY = currentPiece.y + dy;

    if (isValidPosition(board, currentPiece.type, newX, newY, currentPiece.rotation)) {
        currentPiece.x = newX;
        currentPiece.y = newY;
        if (isLocking && (dx !== 0 || dy < 0)) {
            lockCounter = 0;
        }
        lastAction = 'movement';
        lastKickOffset = null;
        return true;
    }
    return false;
}

function rotatePiece() {
    if (!currentPiece) return false;

    const newRotation = (currentPiece.rotation + 1) % 4;
    const kicks = getWallKicks(currentPiece.type, currentPiece.rotation, newRotation);

    for (const kick of kicks) {
        const newX = currentPiece.x + kick[0];
        const newY = currentPiece.y - kick[1];

        if (isValidPosition(board, currentPiece.type, newX, newY, newRotation)) {
            currentPiece.x = newX;
            currentPiece.y = newY;
            currentPiece.rotation = newRotation;
            if (isLocking) {
                lockCounter = 0;
            }
            lastAction = 'rotation';
            lastKickOffset = kick;
            return true;
        }
    }
    return false;
}

function hardDrop() {
    if (!currentPiece) return;
    while (movePiece(0, 1)) {}
    lastAction = 'drop';
    lastKickOffset = null;
    lockPieceToBoard();
}

function getTSpinCorners(piece) {
    const cx = piece.x + 1;
    const cy = piece.y + 1;
    const corners = [
        [cx - 1, cy - 1],
        [cx + 1, cy - 1],
        [cx - 1, cy + 1],
        [cx + 1, cy + 1]
    ];
    const frontCornersMap = [
        [0, 1],
        [1, 3],
        [2, 3],
        [0, 2]
    ];
    return {
        corners: corners,
        frontIndices: frontCornersMap[piece.rotation]
    };
}

function detectTSpin(piece, board, lastAction, lastKickOffset) {
    if (piece.type !== 'T') return null;
    if (lastAction !== 'rotation') return null;
    const { corners, frontIndices } = getTSpinCorners(piece);
    let occupiedCount = 0;
    let frontOccupiedCount = 0;
    corners.forEach(function(corner, idx) {
        const x = corner[0];
        const y = corner[1];
        const isOccupied = (x < 0 || x >= COLS || y >= board.length || (y >= 0 && board[y][x]));
        if (isOccupied) {
            occupiedCount++;
            if (frontIndices.includes(idx)) {
                frontOccupiedCount++;
            }
        }
    });
    if (occupiedCount < 3) return null;
    const isMini = frontOccupiedCount < 2;
    const isWallKickUpgrade = lastKickOffset && (Math.abs(lastKickOffset[0]) + Math.abs(lastKickOffset[1]) === 3);
    if (isMini && !isWallKickUpgrade) {
        return 'mini';
    }
    return 'full';
}

function lockPieceToBoard() {
    if (!currentPiece) return;

    updatePiecePlaced();
    board = lockPiece(board, currentPiece);
    const lines = checkLines(board);

    if (lines.length > 0) {
        combo++;
        updateComboStats(combo);
        const isDifficultClear = lines.length === 4;
        const hasB2bBonus = b2bActive && isDifficultClear;
        pendingScoreCalc = {
            linesCleared: lines.length,
            comboValue: combo,
            hasB2bBonus: hasB2bBonus
        };
        b2bActive = isDifficultClear;
        clearingLines = lines;
        clearingTimer = 100;
        currentPiece = null;
    } else {
        combo = 0;
        spawnPiece();
    }
}

function processInput() {
    const input = getInput();

    if (input.pause) {
        togglePause();
        return;
    }

    if (gameState === GameState.FROZEN) return;
    if (gameState === GameState.PAUSED) return;
    if (gameOver || clearingLines.length > 0) return;
    if (!currentPiece) return;

    if (input.hold) {
        var couldHold = canHold && currentPiece;
        holdPiece();
        if (couldHold) {
            trackAction();
        }
    }
    if (input.left) {
        if (movePiece(-1, 0)) {
            trackAction();
        }
    }
    if (input.right) {
        if (movePiece(1, 0)) {
            trackAction();
        }
    }
    if (input.down) {
        if (movePiece(0, 1)) {
            dropCounter = 0;
            trackAction();
        }
    }
    if (input.rotate) {
        if (rotatePiece()) {
            trackAction();
        }
    }
    if (input.hardDrop) {
        hardDrop();
        trackAction();
    }
}

function update(deltaTime) {
    if (gameState === GameState.GAME_OVER) return;
    if (gameState === GameState.PAUSED) return;

    cycleTimer += deltaTime;
    if (gameState === GameState.PLAYING && cycleTimer >= PLAY_DURATION) {
        gameState = GameState.FROZEN;
        cycleTimer = 0;
    } else if (gameState === GameState.FROZEN && cycleTimer >= FREEZE_DURATION) {
        gameState = GameState.PLAYING;
        cycleTimer = 0;
    }

    growthTimer += deltaTime;
    if (growthTimer >= boardGrowthInterval && board.length < MAX_ROWS) {
        board = growBoard(board, COLS);
        resizeCanvas(board);
        growthTimer = 0;
    }

    if (gameState === GameState.FROZEN) return;

    if (clearingLines.length > 0) {
        clearingTimer -= deltaTime;
        if (clearingTimer <= 0) {
            const result = clearLines(board, clearingLines);
            board = result.board;
            let baseScore = result.linesCleared * pointsPerRow;
            if (pendingScoreCalc && pendingScoreCalc.hasB2bBonus && result.linesCleared === 4) {
                baseScore = Math.floor(baseScore * 1.5);
                incrementB2bCount();
            }
            let comboBonus = 0;
            if (pendingScoreCalc && pendingScoreCalc.comboValue > 1) {
                comboBonus = 50 * (pendingScoreCalc.comboValue - 1) * level;
            }
            score += baseScore + comboBonus;
            pendingScoreCalc = null;
            stats.score = score;
            stats.lines += result.linesCleared;
            checkLevelUp();
            clearingLines = [];
            spawnPiece();
        }
        return;
    }

    if (!currentPiece) return;

    dropCounter += deltaTime;
    if (dropCounter > dropInterval) {
        if (!movePiece(0, 1)) {
            if (!isLocking) {
                isLocking = true;
                lockCounter = 0;
            }
        } else {
            isLocking = false;
            lockCounter = 0;
        }
        dropCounter = 0;
    }

    if (isLocking) {
        lockCounter += deltaTime;
        if (lockCounter >= lockDelay) {
            lockPieceToBoard();
        }
    }
}

function render() {
    drawGrid(board);
    drawBoard(board);

    if (clearingLines.length > 0) {
        drawClearingLines(clearingLines);
    }

    if (currentPiece) {
        const ghostY = getGhostY(currentPiece);
        if (ghostY !== currentPiece.y) {
            drawGhost(currentPiece.type, currentPiece.x, ghostY, currentPiece.rotation);
        }
        drawPiece(currentPiece.type, currentPiece.x, currentPiece.y, currentPiece.rotation);
    }

    drawSidebar(board);
    drawSessionStats();
    drawComboIndicator(combo, b2bActive);
    drawScore(score, level);
    drawNextPreview(nextPiece);
    drawHoldPreview(heldPiece, canHold);

    if (gameState === GameState.FROZEN) {
        drawFreezeOverlay(FREEZE_DURATION - cycleTimer, board);
    }

    if (gameState === GameState.PAUSED) {
        drawPaused(board);
    }

    if (gameState === GameState.GAME_OVER) {
        drawSessionSummary(board, score, level);
    }
}

function resetGame() {
    board = createBoard();
    resizeCanvas(board);
    bag = [];
    nextPiece = null;
    heldPiece = null;
    canHold = true;
    score = 0;
    level = 1;
    gameOver = false;
    gameState = GameState.PLAYING;
    cycleTimer = 0;
    growthTimer = 0;
    clearingLines = [];
    clearingTimer = 0;
    dropCounter = 0;
    lockCounter = 0;
    isLocking = false;
    combo = 0;
    b2bActive = false;
    pendingScoreCalc = null;
    lastAction = null;
    lastKickOffset = null;
    spawnPiece();
    startSession();
}

document.addEventListener('DOMContentLoaded', function() {
    setupCanvas();
    setupInput();
    board = createBoard();
    spawnPiece();
    startSession();

    channel.onmessage = function(event) {
        var data = event.data;
        var type = data.type;
        var payload = data.payload;

        if (type === 'THEME_CHANGE') {
            applyTheme(payload.themeName);
        }
        else if (type === 'SPEED_CHANGE') {
            dropInterval = payload.dropInterval;
        }
        else if (type === 'POINTS_CHANGE') {
            pointsPerRow = payload.pointsPerRow;
        }
        else if (type === 'GROWTH_INTERVAL_CHANGE') {
            boardGrowthInterval = payload.interval;
            growthTimer = Math.min(growthTimer, boardGrowthInterval);
        }
        else if (type === 'STATS_REQUEST') {
            var currentThemeName = 'classic';
            for (var key in THEMES) {
                if (THEMES[key] === currentTheme) {
                    currentThemeName = key;
                    break;
                }
            }
            sendMessage('STATS_RESPONSE', {
                score: score,
                level: level,
                theme: currentThemeName,
                paused: gameState === GameState.PAUSED,
                gameOver: gameState === GameState.GAME_OVER
            });
        }
    };

    document.addEventListener('keydown', function(e) {
        if (gameOver && e.code === 'KeyR') {
            resetGame();
        }
    });

    function gameLoop(timestamp) {
        const deltaTime = timestamp - lastTime;
        lastTime = timestamp;

        processInput();
        update(deltaTime);
        render();

        requestAnimationFrame(gameLoop);
    }

    requestAnimationFrame(gameLoop);
});
