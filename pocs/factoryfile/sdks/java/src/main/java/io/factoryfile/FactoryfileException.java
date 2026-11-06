package io.factoryfile;

public class FactoryfileException extends RuntimeException {
    private final int line;
    private final int col;

    public FactoryfileException(int line, int col, String message) {
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
