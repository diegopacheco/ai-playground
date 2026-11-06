from dataclasses import dataclass
from enum import StrEnum

from agentsfile.errors import AgentsfileError


class TokenKind(StrEnum):
    IDENTIFIER = "identifier"
    STRING = "string"
    EQUALS = "'='"
    LBRACE = "'{'"
    RBRACE = "'}'"
    LBRACKET = "'['"
    RBRACKET = "']'"
    COMMA = "','"
    EOF = "end of file"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    line: int
    col: int


PUNCTUATION = {
    "=": TokenKind.EQUALS,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    ",": TokenKind.COMMA,
}

ESCAPES = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}

WHITESPACE = " \t\r\n"


def _is_identifier_start(char: str) -> bool:
    return char == "_" or ("a" <= char <= "z") or ("A" <= char <= "Z")


def _is_identifier_part(char: str) -> bool:
    return _is_identifier_start(char) or char == "-" or ("0" <= char <= "9")


class _Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.line = 1
        self.col = 1

    def peek(self) -> str:
        return self.text[self.index] if self.index < len(self.text) else ""

    def advance(self) -> str:
        char = self.text[self.index]
        self.index += 1
        if char == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return char

    def tokens(self) -> list[Token]:
        result: list[Token] = []
        while True:
            char = self.peek()
            if char == "":
                result.append(Token(TokenKind.EOF, "", self.line, self.col))
                return result
            if char in WHITESPACE:
                self.advance()
            elif char == "#":
                while self.peek() not in ("", "\n"):
                    self.advance()
            elif char in PUNCTUATION:
                result.append(Token(PUNCTUATION[char], char, self.line, self.col))
                self.advance()
            elif char == '"':
                result.append(self.string())
            elif _is_identifier_start(char):
                result.append(self.identifier())
            else:
                raise AgentsfileError(self.line, self.col, f"unexpected character '{char}'")

    def identifier(self) -> Token:
        line, col, start = self.line, self.col, self.index
        while _is_identifier_part(self.peek()):
            self.advance()
        return Token(TokenKind.IDENTIFIER, self.text[start : self.index], line, col)

    def string(self) -> Token:
        line, col = self.line, self.col
        self.advance()
        chars: list[str] = []
        while True:
            char = self.peek()
            if char in ("", "\n"):
                raise AgentsfileError(line, col, "unterminated string")
            if char == '"':
                self.advance()
                return Token(TokenKind.STRING, "".join(chars), line, col)
            if char == "\\":
                escape_line, escape_col = self.line, self.col
                self.advance()
                code = self.peek()
                if code in ("", "\n"):
                    raise AgentsfileError(line, col, "unterminated string")
                if code not in ESCAPES:
                    raise AgentsfileError(escape_line, escape_col, f"invalid escape '\\{code}'")
                chars.append(ESCAPES[code])
                self.advance()
            else:
                chars.append(self.advance())


def tokenize(text: str) -> list[Token]:
    return _Lexer(text).tokens()
