from dataclasses import dataclass
from fractions import Fraction

from calculator_app.tokens import CalculatorError, Token, tokenize


@dataclass(frozen=True, slots=True)
class Number:
    value: Fraction


@dataclass(frozen=True, slots=True)
class Negate:
    operand: Node


@dataclass(frozen=True, slots=True)
class Binary:
    operator: str
    left: Node
    right: Node


type Node = Number | Negate | Binary

PRECEDENCE: dict[str, int] = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3}
RIGHT_ASSOCIATIVE = frozenset({"^"})
UNARY_PRECEDENCE = 3


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def advance(self) -> Token:
        token = self.peek()
        if token is None:
            raise CalculatorError("unexpected end of expression")
        self.index += 1
        return token

    def parse(self) -> Node:
        node = self.expression(0)
        leftover = self.peek()
        if leftover is not None:
            raise CalculatorError(f"unexpected '{leftover.text}' at position {leftover.position}")
        return node

    def expression(self, minimum: int) -> Node:
        left = self.unary()
        while (token := self.peek()) is not None and token.kind == "operator":
            precedence = PRECEDENCE[token.text]
            if precedence < minimum:
                break
            self.advance()
            next_minimum = precedence if token.text in RIGHT_ASSOCIATIVE else precedence + 1
            left = Binary(token.text, left, self.expression(next_minimum))
        return left

    def unary(self) -> Node:
        token = self.peek()
        if token is not None and token.kind == "operator" and token.text in {"+", "-"}:
            self.advance()
            operand = self.expression(UNARY_PRECEDENCE)
            return Negate(operand) if token.text == "-" else operand
        return self.primary()

    def primary(self) -> Node:
        token = self.advance()
        match token.kind:
            case "number" if token.value is not None:
                return Number(token.value)
            case "lparen":
                node = self.expression(0)
                closing = self.advance()
                if closing.kind != "rparen":
                    raise CalculatorError(f"expected ')' at position {closing.position}")
                return node
            case _:
                raise CalculatorError(f"unexpected '{token.text}' at position {token.position}")


def parse(source: str) -> Node:
    return Parser(tokenize(source)).parse()
