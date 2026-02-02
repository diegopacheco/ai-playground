let board;

document.addEventListener('DOMContentLoaded', function() {
    setupCanvas();
    board = createBoard();
    drawGrid();
    drawBoard(board);
});
