let canvas;
let ctx;
const SIDEBAR_WIDTH = 120;

function setupCanvas() {
    canvas = document.getElementById('game');
    ctx = canvas.getContext('2d');

    const dpr = window.devicePixelRatio || 1;
    const width = COLS * CELL_SIZE + SIDEBAR_WIDTH;
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

function drawSidebar() {
    const sidebarX = COLS * CELL_SIZE;
    ctx.fillStyle = '#16162a';
    ctx.fillRect(sidebarX, 0, SIDEBAR_WIDTH, ROWS * CELL_SIZE);

    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sidebarX, 0);
    ctx.lineTo(sidebarX, ROWS * CELL_SIZE);
    ctx.stroke();
}

function drawScore(score, level) {
    const sidebarX = COLS * CELL_SIZE;
    const centerX = sidebarX + SIDEBAR_WIDTH / 2;

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    ctx.fillText('SCORE', centerX, 320);

    ctx.font = 'bold 24px Arial';
    ctx.fillStyle = '#00ffff';
    ctx.fillText(score.toString(), centerX, 345);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Arial';
    ctx.fillText('LEVEL', centerX, 400);

    ctx.font = 'bold 24px Arial';
    ctx.fillStyle = '#ffff00';
    ctx.fillText(level.toString(), centerX, 425);
}

function drawGhost(pieceType, x, ghostY, rotation) {
    const piece = PIECES[pieceType];
    const shape = piece.shapes[rotation];

    ctx.globalAlpha = 0.3;
    ctx.fillStyle = piece.color;

    for (let row = 0; row < shape.length; row++) {
        for (let col = 0; col < shape[row].length; col++) {
            if (shape[row][col]) {
                const drawX = (x + col) * CELL_SIZE + 1;
                const drawY = (ghostY + row) * CELL_SIZE + 1;
                if (ghostY + row >= 0) {
                    ctx.fillRect(drawX, drawY, CELL_SIZE - 2, CELL_SIZE - 2);
                }
            }
        }
    }

    ctx.globalAlpha = 1.0;
}

function drawNextPreview(pieceType) {
    const sidebarX = COLS * CELL_SIZE;
    const centerX = sidebarX + SIDEBAR_WIDTH / 2;

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('NEXT', centerX, 20);

    const boxX = sidebarX + 10;
    const boxY = 45;
    const boxSize = SIDEBAR_WIDTH - 20;

    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 2;
    ctx.strokeRect(boxX, boxY, boxSize, boxSize);

    if (pieceType) {
        const piece = PIECES[pieceType];
        const shape = piece.shapes[0];
        const cellSize = 20;

        const pieceWidth = shape[0].length * cellSize;
        const pieceHeight = shape.length * cellSize;
        const offsetX = boxX + (boxSize - pieceWidth) / 2;
        const offsetY = boxY + (boxSize - pieceHeight) / 2;

        ctx.fillStyle = piece.color;
        for (let row = 0; row < shape.length; row++) {
            for (let col = 0; col < shape[row].length; col++) {
                if (shape[row][col]) {
                    ctx.fillRect(
                        offsetX + col * cellSize + 1,
                        offsetY + row * cellSize + 1,
                        cellSize - 2,
                        cellSize - 2
                    );
                }
            }
        }
    }
}

function drawHoldPreview(pieceType, canHold) {
    const sidebarX = COLS * CELL_SIZE;
    const centerX = sidebarX + SIDEBAR_WIDTH / 2;

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('HOLD', centerX, 160);

    const boxX = sidebarX + 10;
    const boxY = 185;
    const boxSize = SIDEBAR_WIDTH - 20;

    ctx.strokeStyle = canHold ? '#2a2a4a' : '#1a1a2a';
    ctx.lineWidth = 2;
    ctx.strokeRect(boxX, boxY, boxSize, boxSize);

    if (pieceType) {
        const piece = PIECES[pieceType];
        const shape = piece.shapes[0];
        const cellSize = 20;

        const pieceWidth = shape[0].length * cellSize;
        const pieceHeight = shape.length * cellSize;
        const offsetX = boxX + (boxSize - pieceWidth) / 2;
        const offsetY = boxY + (boxSize - pieceHeight) / 2;

        ctx.globalAlpha = canHold ? 1.0 : 0.4;
        ctx.fillStyle = piece.color;
        for (let row = 0; row < shape.length; row++) {
            for (let col = 0; col < shape[row].length; col++) {
                if (shape[row][col]) {
                    ctx.fillRect(
                        offsetX + col * cellSize + 1,
                        offsetY + row * cellSize + 1,
                        cellSize - 2,
                        cellSize - 2
                    );
                }
            }
        }
        ctx.globalAlpha = 1.0;
    }
}

function drawPaused() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 36px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('PAUSED', (COLS * CELL_SIZE) / 2, (ROWS * CELL_SIZE) / 2 - 20);

    ctx.font = '18px Arial';
    ctx.fillText('Press P to resume', (COLS * CELL_SIZE) / 2, (ROWS * CELL_SIZE) / 2 + 20);
}
