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
let isPaused = false;
let score = 0;
let level = 1;
let clearingLines = [];
let clearingTimer = 0;

function calculateLevel() {
    return Math.floor(score / 100) + 1;
}

function checkLevelUp() {
    const newLevel = calculateLevel();
    if (newLevel > level) {
        level = newLevel;
        onLevelUp();
    }
}

function onLevelUp() {
    console.log('Level up to ' + level);
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
    isPaused = !isPaused;
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
            return true;
        }
    }
    return false;
}

function hardDrop() {
    if (!currentPiece) return;
    while (movePiece(0, 1)) {}
    lockPieceToBoard();
}

function lockPieceToBoard() {
    if (!currentPiece) return;

    board = lockPiece(board, currentPiece);
    const lines = checkLines(board);

    if (lines.length > 0) {
        clearingLines = lines;
        clearingTimer = 100;
        currentPiece = null;
    } else {
        spawnPiece();
    }
}

function processInput() {
    const input = getInput();

    if (input.pause) {
        togglePause();
        return;
    }

    if (isPaused) return;
    if (gameOver || clearingLines.length > 0) return;
    if (!currentPiece) return;

    if (input.hold) {
        holdPiece();
    }
    if (input.left) {
        movePiece(-1, 0);
    }
    if (input.right) {
        movePiece(1, 0);
    }
    if (input.down) {
        if (movePiece(0, 1)) {
            dropCounter = 0;
        }
    }
    if (input.rotate) {
        rotatePiece();
    }
    if (input.hardDrop) {
        hardDrop();
    }
}

function update(deltaTime) {
    if (gameOver) return;
    if (isPaused) return;

    if (clearingLines.length > 0) {
        clearingTimer -= deltaTime;
        if (clearingTimer <= 0) {
            const result = clearLines(board, clearingLines);
            board = result.board;
            score += result.linesCleared * 10;
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
    drawGrid();
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

    drawSidebar();
    drawScore(score, level);
    drawNextPreview(nextPiece);
    drawHoldPreview(heldPiece, canHold);

    if (isPaused) {
        drawPaused();
    }

    if (gameOver) {
        drawGameOver();
    }
}

function resetGame() {
    board = createBoard();
    bag = [];
    nextPiece = null;
    heldPiece = null;
    canHold = true;
    score = 0;
    level = 1;
    gameOver = false;
    isPaused = false;
    clearingLines = [];
    clearingTimer = 0;
    dropCounter = 0;
    lockCounter = 0;
    isLocking = false;
    spawnPiece();
}

document.addEventListener('DOMContentLoaded', function() {
    setupCanvas();
    setupInput();
    board = createBoard();
    spawnPiece();

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
