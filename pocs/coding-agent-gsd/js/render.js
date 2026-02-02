let canvas;
let ctx;

function setupCanvas() {
    canvas = document.getElementById('game');
    ctx = canvas.getContext('2d');

    const dpr = window.devicePixelRatio || 1;
    const width = COLS * CELL_SIZE;
    const height = ROWS * CELL_SIZE;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    ctx.scale(dpr, dpr);
}

function drawGrid() {
    ctx.fillStyle = '#0f0f1a';
    ctx.fillRect(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE);

    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;

    for (let col = 0; col <= COLS; col++) {
        ctx.beginPath();
        ctx.moveTo(col * CELL_SIZE, 0);
        ctx.lineTo(col * CELL_SIZE, ROWS * CELL_SIZE);
        ctx.stroke();
    }

    for (let row = 0; row <= ROWS; row++) {
        ctx.beginPath();
        ctx.moveTo(0, row * CELL_SIZE);
        ctx.lineTo(COLS * CELL_SIZE, row * CELL_SIZE);
        ctx.stroke();
    }
}

function drawBoard(board) {
    for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
            if (board[row][col]) {
                ctx.fillStyle = board[row][col];
                ctx.fillRect(
                    col * CELL_SIZE + 1,
                    row * CELL_SIZE + 1,
                    CELL_SIZE - 2,
                    CELL_SIZE - 2
                );
            }
        }
    }
}

function drawPiece(pieceType, x, y, rotation) {
    const piece = PIECES[pieceType];
    const shape = piece.shapes[rotation];
    ctx.fillStyle = piece.color;

    for (let row = 0; row < shape.length; row++) {
        for (let col = 0; col < shape[row].length; col++) {
            if (shape[row][col]) {
                const drawX = (x + col) * CELL_SIZE + 1;
                const drawY = (y + row) * CELL_SIZE + 1;
                if (y + row >= 0) {
                    ctx.fillRect(drawX, drawY, CELL_SIZE - 2, CELL_SIZE - 2);
                }
            }
        }
    }
}

function drawClearingLines(lines) {
    ctx.fillStyle = '#ffffff';
    for (const row of lines) {
        ctx.fillRect(0, row * CELL_SIZE, COLS * CELL_SIZE, CELL_SIZE);
    }
}

function drawGameOver() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE);

    ctx.fillStyle = '#ff0000';
    ctx.font = 'bold 36px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('GAME OVER', (COLS * CELL_SIZE) / 2, (ROWS * CELL_SIZE) / 2 - 20);

    ctx.fillStyle = '#ffffff';
    ctx.font = '18px Arial';
    ctx.fillText('Press R to restart', (COLS * CELL_SIZE) / 2, (ROWS * CELL_SIZE) / 2 + 20);
}
