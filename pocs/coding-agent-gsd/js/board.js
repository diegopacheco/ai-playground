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
