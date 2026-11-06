package io.factoryfile;

import io.factoryfile.Statement.Attribute;
import io.factoryfile.Statement.Block;
import io.factoryfile.Value.ListValue;
import io.factoryfile.Value.MapValue;
import io.factoryfile.Value.StringValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Parser {
    private final List<Token> tokens;
    private int pos;

    public Parser(List<Token> tokens) {
        this.tokens = List.copyOf(tokens);
    }

    public List<Statement> parse() {
        List<Statement> statements = new ArrayList<>();
        while (peek().kind() != TokenKind.EOF) {
            statements.add(statement("expected identifier"));
        }
        return statements;
    }

    private Statement statement(String expectation) {
        Token name = expect(TokenKind.IDENTIFIER, expectation);
        Token next = peek();
        return switch (next.kind()) {
            case EQUALS -> {
                pos++;
                yield new Attribute(name.text(), value(), name.line(), name.col());
            }
            case STRING -> {
                pos++;
                expect(TokenKind.LBRACE, "expected '{'");
                yield new Block(name.text(), next.text(), body(), name.line(), name.col());
            }
            default -> throw error(next, "expected '=' or block label");
        };
    }

    private List<Statement> body() {
        List<Statement> statements = new ArrayList<>();
        while (peek().kind() != TokenKind.RBRACE) {
            statements.add(statement("expected identifier or '}'"));
        }
        pos++;
        return statements;
    }

    private Value value() {
        Token start = peek();
        return switch (start.kind()) {
            case STRING -> string("expected value");
            case LBRACKET -> list();
            case LBRACE -> map();
            default -> throw error(start, "expected value");
        };
    }

    private ListValue list() {
        Token open = expect(TokenKind.LBRACKET, "expected '['");
        List<StringValue> items = new ArrayList<>();
        while (peek().kind() != TokenKind.RBRACKET) {
            items.add(string("expected string"));
            if (peek().kind() == TokenKind.COMMA) {
                pos++;
            } else if (peek().kind() != TokenKind.RBRACKET) {
                throw error(peek(), "expected ',' or ']'");
            }
        }
        pos++;
        return new ListValue(items, open.line(), open.col());
    }

    private MapValue map() {
        Token open = expect(TokenKind.LBRACE, "expected '{'");
        Map<String, StringValue> entries = new LinkedHashMap<>();
        while (peek().kind() != TokenKind.RBRACE) {
            Token key = expect(TokenKind.IDENTIFIER, "expected identifier or '}'");
            if (entries.containsKey(key.text())) {
                throw new FactoryfileException(key.line(), key.col(), "duplicate key \"" + key.text() + "\"");
            }
            expect(TokenKind.EQUALS, "expected '='");
            entries.put(key.text(), string("expected string"));
            if (peek().kind() == TokenKind.COMMA) {
                pos++;
            }
        }
        pos++;
        return new MapValue(entries, open.line(), open.col());
    }

    private StringValue string(String expectation) {
        Token token = expect(TokenKind.STRING, expectation);
        return new StringValue(token.text(), token.line(), token.col());
    }

    private Token expect(TokenKind kind, String expectation) {
        Token token = peek();
        if (token.kind() != kind) {
            throw error(token, expectation);
        }
        pos++;
        return token;
    }

    private Token peek() {
        return tokens.get(pos);
    }

    private static FactoryfileException error(Token token, String expectation) {
        return new FactoryfileException(token.line(), token.col(), expectation + ", found " + token.kind().label());
    }
}
