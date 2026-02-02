let board;
let currentPiece = null;
let bag = [];

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
        y: -1,
        rotation: 0
    };
}

function movePiece(dx, dy) {
    const newX = currentPiece.x + dx;
    const newY = currentPiece.y + dy;

    if (isValidPosition(board, currentPiece.type, newX, newY, currentPiece.rotation)) {
        currentPiece.x = newX;
        currentPiece.y = newY;
        return true;
    }
    return false;
}

function rotatePiece() {
    const newRotation = (currentPiece.rotation + 1) % 4;
    const kicks = getWallKicks(currentPiece.type, currentPiece.rotation, newRotation);

    for (const kick of kicks) {
        const newX = currentPiece.x + kick[0];
        const newY = currentPiece.y - kick[1];

        if (isValidPosition(board, currentPiece.type, newX, newY, newRotation)) {
            currentPiece.x = newX;
            currentPiece.y = newY;
            currentPiece.rotation = newRotation;
            return true;
        }
    }
    return false;
}

function hardDrop() {
    while (movePiece(0, 1)) {}
}

function processInput() {
    if (!currentPiece) return;

    const input = getInput();

    if (input.left) {
        movePiece(-1, 0);
    }
    if (input.right) {
        movePiece(1, 0);
    }
    if (input.rotate) {
        rotatePiece();
    }
    if (input.hardDrop) {
        hardDrop();
    }
}

function render() {
    drawGrid();
    drawBoard(board);
    if (currentPiece) {
        drawPiece(currentPiece.type, currentPiece.x, currentPiece.y, currentPiece.rotation);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    setupCanvas();
    setupInput();
    board = createBoard();
    spawnPiece();
    render();

    function gameLoop() {
        processInput();
        render();
        requestAnimationFrame(gameLoop);
    }
    requestAnimationFrame(gameLoop);
});
