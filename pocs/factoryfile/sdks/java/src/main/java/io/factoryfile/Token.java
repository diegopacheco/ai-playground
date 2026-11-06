package io.factoryfile;

public record Token(TokenKind kind, String text, int line, int col) {
}
