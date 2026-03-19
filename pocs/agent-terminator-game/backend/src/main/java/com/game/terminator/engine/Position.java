package com.game.terminator.engine;

public record Position(int x, int y) {
    public Position move(Direction dir, int gridSize) {
        int nx = x + dir.dx;
        int ny = y + dir.dy;
        nx = Math.max(0, Math.min(gridSize - 1, nx));
        ny = Math.max(0, Math.min(gridSize - 1, ny));
        return new Position(nx, ny);
    }
}
