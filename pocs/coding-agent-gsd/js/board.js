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
