let board;
let currentPiece = null;
let bag = [];
let lastTime = 0;
let dropCounter = 0;
let dropInterval = 1000;
let lockCounter = 0;
let lockDelay = 500;
let isLocking = false;
let gameOver = false;
let score = 0;
let clearingLines = [];
let clearingTimer = 0;

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
    const type = getNextPiece();
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
    if (gameOver || clearingLines.length > 0) return;
    if (!currentPiece) return;

    const input = getInput();

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

    if (clearingLines.length > 0) {
        clearingTimer -= deltaTime;
        if (clearingTimer <= 0) {
            const result = clearLines(board, clearingLines);
            board = result.board;
            score += result.linesCleared * 10;
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
        drawPiece(currentPiece.type, currentPiece.x, currentPiece.y, currentPiece.rotation);
    }

    if (gameOver) {
        drawGameOver();
    }
}

function resetGame() {
    board = createBoard();
    bag = [];
    score = 0;
    gameOver = false;
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
