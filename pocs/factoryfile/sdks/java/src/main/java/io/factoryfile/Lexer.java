package io.factoryfile;

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
        if (c == '<' && pos + 1 < chars.length && chars[pos + 1] == '<') {
            return heredoc(startLine, startCol);
        }
        if (isIdentifierStart(c)) {
            return identifier(startLine, startCol);
        }
        throw new FactoryfileException(startLine, startCol, "unexpected character '" + Character.toString(c) + "'");
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
                throw new FactoryfileException(startLine, startCol, "unterminated string");
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

    private Token heredoc(int startLine, int startCol) {
        advance();
        advance();
        StringBuilder marker = new StringBuilder();
        while (pos < chars.length && isMarkerPart(chars[pos])) {
            marker.appendCodePoint(chars[pos]);
            advance();
        }
        if (marker.isEmpty() || !isIdentifierStart(marker.charAt(0))) {
            throw new FactoryfileException(startLine, startCol, "invalid heredoc marker");
        }
        while (pos < chars.length && (chars[pos] == ' ' || chars[pos] == '\t' || chars[pos] == '\r')) {
            advance();
        }
        if (pos >= chars.length) {
            throw new FactoryfileException(startLine, startCol, "unterminated heredoc");
        }
        if (chars[pos] != '\n') {
            throw new FactoryfileException(startLine, startCol, "invalid heredoc marker");
        }
        advance();
        List<String> lines = new ArrayList<>();
        while (pos < chars.length) {
            String text = restOfLine();
            if (stripBlanks(text).equals(marker.toString())) {
                return new Token(TokenKind.STRING, Heredoc.dedent(lines), startLine, startCol);
            }
            lines.add(text);
        }
        throw new FactoryfileException(startLine, startCol, "unterminated heredoc");
    }

    private String restOfLine() {
        StringBuilder text = new StringBuilder();
        while (pos < chars.length && chars[pos] != '\n') {
            text.appendCodePoint(chars[pos]);
            advance();
        }
        if (pos < chars.length) {
            advance();
        }
        return text.toString();
    }

    private static String stripBlanks(String text) {
        int start = 0;
        int end = text.length();
        while (start < end && Heredoc.isBlankChar(text.charAt(start))) {
            start++;
        }
        while (end > start && Heredoc.isBlankChar(text.charAt(end - 1))) {
            end--;
        }
        return text.substring(start, end);
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
            default -> throw new FactoryfileException(escapeLine, escapeCol, "invalid escape '\\" + Character.toString(c) + "'");
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

    private static boolean isMarkerPart(int c) {
        return isIdentifierStart(c) || (c >= '0' && c <= '9');
    }

    private static boolean isIdentifierPart(int c) {
        return isIdentifierStart(c) || (c >= '0' && c <= '9') || c == '-';
    }
}
