const COLS = 10;
const ROWS = 20;
const CELL_SIZE = 30;

function createBoard() {
    const board = [];
    for (let row = 0; row < ROWS; row++) {
        board[row] = [];
        for (let col = 0; col < COLS; col++) {
            board[row][col] = null;
        }
    }
    return board;
}

function isValidPosition(board, pieceType, x, y, rotation) {
    const shape = PIECES[pieceType].shapes[rotation];
    for (let row = 0; row < shape.length; row++) {
        for (let col = 0; col < shape[row].length; col++) {
            if (shape[row][col]) {
                const newX = x + col;
                const newY = y + row;
                if (newX < 0 || newX >= COLS || newY >= ROWS) {
                    return false;
                }
                if (newY >= 0 && board[newY][newX]) {
                    return false;
                }
            }
        }
    }
    return true;
}

function lockPiece(board, piece) {
    const shape = PIECES[piece.type].shapes[piece.rotation];

    for (let row = 0; row < shape.length; row++) {
        for (let col = 0; col < shape[row].length; col++) {
            if (shape[row][col]) {
                const boardY = piece.y + row;
                const boardX = piece.x + col;
                if (boardY >= 0 && boardY < ROWS && boardX >= 0 && boardX < COLS) {
                    board[boardY][boardX] = piece.type;
                }
            }
        }
    }
    return board;
}

function checkLines(board) {
    const fullLines = [];
    for (let row = 0; row < ROWS; row++) {
        let isFull = true;
        for (let col = 0; col < COLS; col++) {
            if (!board[row][col]) {
                isFull = false;
                break;
            }
        }
        if (isFull) {
            fullLines.push(row);
        }
    }
    return fullLines;
}

function clearLines(board, lines) {
    const sortedLines = [...lines].sort((a, b) => a - b);

    for (let i = sortedLines.length - 1; i >= 0; i--) {
        board.splice(sortedLines[i], 1);
    }

    for (let i = 0; i < sortedLines.length; i++) {
        const newRow = [];
        for (let col = 0; col < COLS; col++) {
            newRow.push(null);
        }
        board.unshift(newRow);
    }

    return { board: board, linesCleared: sortedLines.length };
}
