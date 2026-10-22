import re
from fractions import Fraction

from calculator.arithmetic import CalculatorError, apply, to_fraction

TOKEN = re.compile(r"\s*(?:(\d+\.?\d*|\.\d+)|(.))")


def tokenize(expression: str) -> list[str]:
    tokens: list[str] = []
    for number, symbol in TOKEN.findall(expression.strip()):
        if symbol and symbol not in "+-*/()":
            raise CalculatorError(f"unexpected character: {symbol!r}")
        tokens.append(number or symbol)
    if not tokens:
        raise CalculatorError("empty expression")
    return tokens


class Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise CalculatorError("unexpected end of expression")
        self.position += 1
        return token

    def parse(self) -> Fraction:
        value = self.expression()
        if self.peek() is not None:
            raise CalculatorError(f"unexpected token: {self.peek()!r}")
        return value

    def expression(self) -> Fraction:
        value = self.term()
        while self.peek() in ("+", "-"):
            value = apply(self.take(), value, self.term())
        return value

    def term(self) -> Fraction:
        value = self.factor()
        while self.peek() in ("*", "/"):
            value = apply(self.take(), value, self.factor())
        return value

    def factor(self) -> Fraction:
        token = self.take()
        match token:
            case "-":
                return -self.factor()
            case "+":
                return self.factor()
            case "(":
                value = self.expression()
                if self.take() != ")":
                    raise CalculatorError("missing closing parenthesis")
                return value
            case "*" | "/" | ")":
                raise CalculatorError(f"unexpected token: {token!r}")
            case _:
                return to_fraction(token)


def evaluate_infix(expression: str) -> Fraction:
    return Parser(tokenize(expression)).parse()
