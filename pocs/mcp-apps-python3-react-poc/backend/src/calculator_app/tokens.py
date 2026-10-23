from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Literal

type TokenKind = Literal["number", "operator", "lparen", "rparen"]

OPERATOR_ALIASES: dict[str, str] = {
    "+": "+",
    "-": "-",
    "−": "-",
    "*": "*",
    "×": "*",
    "x": "*",
    "/": "/",
    "÷": "/",
    "%": "%",
    "^": "^",
}


class CalculatorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    text: str
    position: int
    value: Fraction | None = None


def read_number(source: str, start: int) -> tuple[Token, int]:
    end = start
    while end < len(source) and (source[end].isdigit() or source[end] == "."):
        end += 1
    text = source[start:end]
    if text.count(".") > 1 or text == ".":
        raise CalculatorError(f"invalid number '{text}' at position {start}")
    try:
        value = Fraction(Decimal(text))
    except InvalidOperation as error:
        raise CalculatorError(f"invalid number '{text}' at position {start}") from error
    return Token("number", text, start, value), end


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
        elif char.isdigit() or char == ".":
            token, index = read_number(source, index)
            tokens.append(token)
        elif char in OPERATOR_ALIASES:
            tokens.append(Token("operator", OPERATOR_ALIASES[char], index))
            index += 1
        elif char == "(":
            tokens.append(Token("lparen", char, index))
            index += 1
        elif char == ")":
            tokens.append(Token("rparen", char, index))
            index += 1
        else:
            raise CalculatorError(f"unexpected character '{char}' at position {index}")
    if not tokens:
        raise CalculatorError("expression is empty")
    return tokens
