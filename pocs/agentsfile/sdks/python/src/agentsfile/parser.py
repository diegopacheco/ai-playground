from dataclasses import dataclass

from agentsfile.errors import AgentsfileError
from agentsfile.lexer import Token, TokenKind


@dataclass(frozen=True)
class StringValue:
    value: str
    line: int
    col: int


@dataclass(frozen=True)
class ListValue:
    items: tuple[str, ...]
    line: int
    col: int


@dataclass(frozen=True)
class MapValue:
    entries: dict[str, str]
    line: int
    col: int


type Value = StringValue | ListValue | MapValue


@dataclass(frozen=True)
class Attribute:
    name: str
    line: int
    col: int
    value: Value


@dataclass(frozen=True)
class Block:
    kind: str
    label: str
    line: int
    col: int
    body: tuple[Statement, ...]


type Statement = Attribute | Block


def _fail(token: Token, expected: str) -> AgentsfileError:
    return AgentsfileError(token.line, token.col, f"expected {expected}, found {token.kind}")


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> Token:
        return self.tokens[self.index]

    def next(self) -> Token:
        token = self.tokens[self.index]
        if token.kind != TokenKind.EOF:
            self.index += 1
        return token

    def expect(self, kind: TokenKind, expected: str) -> Token:
        token = self.next()
        if token.kind != kind:
            raise _fail(token, expected)
        return token

    def file(self) -> tuple[Statement, ...]:
        statements: list[Statement] = []
        while self.peek().kind != TokenKind.EOF:
            statements.append(self.statement(self.expect(TokenKind.IDENTIFIER, "identifier")))
        return tuple(statements)

    def body(self) -> tuple[Statement, ...]:
        statements: list[Statement] = []
        while True:
            token = self.next()
            if token.kind == TokenKind.RBRACE:
                return tuple(statements)
            if token.kind != TokenKind.IDENTIFIER:
                raise _fail(token, "identifier or '}'")
            statements.append(self.statement(token))

    def statement(self, name: Token) -> Statement:
        token = self.next()
        if token.kind == TokenKind.EQUALS:
            return Attribute(name.text, name.line, name.col, self.value())
        if token.kind == TokenKind.STRING:
            self.expect(TokenKind.LBRACE, "'{'")
            return Block(name.text, token.text, name.line, name.col, self.body())
        raise _fail(token, "'=' or block label")

    def value(self) -> Value:
        token = self.next()
        if token.kind == TokenKind.STRING:
            return StringValue(token.text, token.line, token.col)
        if token.kind == TokenKind.LBRACKET:
            return ListValue(self.list_items(), token.line, token.col)
        if token.kind == TokenKind.LBRACE:
            return MapValue(self.map_entries(), token.line, token.col)
        raise _fail(token, "value")

    def list_items(self) -> tuple[str, ...]:
        items: list[str] = []
        if self.peek().kind == TokenKind.RBRACKET:
            self.next()
            return ()
        while True:
            items.append(self.expect(TokenKind.STRING, "string").text)
            token = self.next()
            if token.kind == TokenKind.RBRACKET:
                return tuple(items)
            if token.kind != TokenKind.COMMA:
                raise _fail(token, "',' or ']'")
            if self.peek().kind == TokenKind.RBRACKET:
                self.next()
                return tuple(items)

    def map_entries(self) -> dict[str, str]:
        entries: dict[str, str] = {}
        while True:
            key = self.next()
            if key.kind == TokenKind.RBRACE:
                return entries
            if key.kind != TokenKind.IDENTIFIER:
                raise _fail(key, "identifier or '}'")
            if key.text in entries:
                raise AgentsfileError(key.line, key.col, f'duplicate key "{key.text}"')
            self.expect(TokenKind.EQUALS, "'='")
            entries[key.text] = self.expect(TokenKind.STRING, "string").text
            if self.peek().kind == TokenKind.COMMA:
                self.next()


def parse_statements(tokens: list[Token]) -> tuple[Statement, ...]:
    return _Parser(tokens).file()
