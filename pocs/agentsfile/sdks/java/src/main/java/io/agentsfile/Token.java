package io.agentsfile;

public record Token(TokenKind kind, String text, int line, int col) {
}
