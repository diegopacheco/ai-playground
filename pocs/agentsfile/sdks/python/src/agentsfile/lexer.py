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

HEREDOC_BLANK = " \t\r"


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
            elif char == "<" and self.text.startswith("<<", self.index):
                result.append(self.heredoc())
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

    def heredoc(self) -> Token:
        line, col = self.line, self.col
        self.advance()
        self.advance()
        start = self.index
        while _is_marker_part(self.peek()):
            self.advance()
        marker = self.text[start : self.index]
        if marker == "" or not _is_identifier_start(marker[0]):
            raise AgentsfileError(line, col, "invalid heredoc marker")
        while self.peek() in (" ", "\t", "\r"):
            self.advance()
        if self.peek() == "":
            raise AgentsfileError(line, col, "unterminated heredoc")
        if self.advance() != "\n":
            raise AgentsfileError(line, col, "invalid heredoc marker")
        lines: list[str] = []
        while self.peek() != "":
            start = self.index
            while self.peek() not in ("", "\n"):
                self.advance()
            text = self.text[start : self.index]
            if self.peek() == "\n":
                self.advance()
            if text.strip(HEREDOC_BLANK) == marker:
                return Token(TokenKind.STRING, _dedent(lines), line, col)
            lines.append(text)
        raise AgentsfileError(line, col, "unterminated heredoc")


def _is_marker_part(char: str) -> bool:
    return char != "" and (char == "_" or (char.isascii() and char.isalnum()))


def _is_blank(text: str) -> bool:
    return text.strip(HEREDOC_BLANK) == ""


def _dedent(lines: list[str]) -> str:
    indent = min((len(text) - len(text.lstrip(" \t")) for text in lines if not _is_blank(text)), default=0)
    return "\n".join("" if _is_blank(text) else text[indent:].rstrip("\r") for text in lines)


def tokenize(text: str) -> list[Token]:
    return _Lexer(text).tokens()
