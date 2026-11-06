package io.agentsfile;

public class AgentsfileException extends RuntimeException {
    private final int line;
    private final int col;

    public AgentsfileException(int line, int col, String message) {
        super("line " + line + ", col " + col + ": " + message);
        this.line = line;
        this.col = col;
    }

    public int line() {
        return line;
    }

    public int col() {
        return col;
    }
}
