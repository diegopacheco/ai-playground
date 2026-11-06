package io.agentsfile;

import java.util.ArrayList;
import java.util.List;

public final class Lexer {
    private final int[] chars;
    private int pos;
    private int line = 1;
    private int col = 1;

    public Lexer(String source) {
        this.chars = source.codePoints().toArray();
    }

    public List<Token> tokenize() {
        List<Token> tokens = new ArrayList<>();
        while (true) {
            skipBlanks();
            if (pos >= chars.length) {
                tokens.add(new Token(TokenKind.EOF, "", line, col));
                return tokens;
            }
            tokens.add(next());
        }
    }

    private void skipBlanks() {
        while (pos < chars.length) {
            int c = chars[pos];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                advance();
            } else if (c == '#') {
                while (pos < chars.length && chars[pos] != '\n') {
                    advance();
                }
            } else {
                return;
            }
        }
    }

    private Token next() {
        int c = chars[pos];
        int startLine = line;
        int startCol = col;
        TokenKind punctuation = punctuation(c);
        if (punctuation != null) {
            advance();
            return new Token(punctuation, Character.toString(c), startLine, startCol);
        }
        if (c == '"') {
            return string(startLine, startCol);
        }
        if (isIdentifierStart(c)) {
            return identifier(startLine, startCol);
        }
        throw new AgentsfileException(startLine, startCol, "unexpected character '" + Character.toString(c) + "'");
    }

    private static TokenKind punctuation(int c) {
        return switch (c) {
            case '=' -> TokenKind.EQUALS;
            case '{' -> TokenKind.LBRACE;
            case '}' -> TokenKind.RBRACE;
            case '[' -> TokenKind.LBRACKET;
            case ']' -> TokenKind.RBRACKET;
            case ',' -> TokenKind.COMMA;
            default -> null;
        };
    }

    private Token identifier(int startLine, int startCol) {
        StringBuilder text = new StringBuilder();
        while (pos < chars.length && isIdentifierPart(chars[pos])) {
            text.appendCodePoint(chars[pos]);
            advance();
        }
        return new Token(TokenKind.IDENTIFIER, text.toString(), startLine, startCol);
    }

    private Token string(int startLine, int startCol) {
        advance();
        StringBuilder text = new StringBuilder();
        while (true) {
            if (pos >= chars.length || chars[pos] == '\n') {
                throw new AgentsfileException(startLine, startCol, "unterminated string");
            }
            int c = chars[pos];
            if (c == '"') {
                advance();
                return new Token(TokenKind.STRING, text.toString(), startLine, startCol);
            }
            if (c == '\\') {
                text.append(escape());
            } else {
                text.appendCodePoint(c);
                advance();
            }
        }
    }

    private char escape() {
        int escapeLine = line;
        int escapeCol = col;
        advance();
        if (pos >= chars.length || chars[pos] == '\n') {
            return '\\';
        }
        int c = chars[pos];
        char decoded = switch (c) {
            case '"' -> '"';
            case '\\' -> '\\';
            case 'n' -> '\n';
            case 't' -> '\t';
            default -> throw new AgentsfileException(escapeLine, escapeCol, "invalid escape '\\" + Character.toString(c) + "'");
        };
        advance();
        return decoded;
    }

    private void advance() {
        if (chars[pos] == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
        pos++;
    }

    private static boolean isIdentifierStart(int c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    }

    private static boolean isIdentifierPart(int c) {
        return isIdentifierStart(c) || (c >= '0' && c <= '9') || c == '-';
    }
}
