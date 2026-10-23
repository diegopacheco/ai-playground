from dataclasses import dataclass, field
from fractions import Fraction

from calculator_app.parser import Binary, Negate, Node, Number
from calculator_app.tokens import CalculatorError

MAX_EXPONENT = 1000
MAX_DECIMALS = 12


def format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    text = f"{float(value):.{MAX_DECIMALS}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def power(base: Fraction, exponent: Fraction) -> Fraction:
    if exponent.denominator != 1:
        raise CalculatorError("exponent must be a whole number")
    if abs(exponent.numerator) > MAX_EXPONENT:
        raise CalculatorError(f"exponent must be between -{MAX_EXPONENT} and {MAX_EXPONENT}")
    if base == 0 and exponent < 0:
        raise CalculatorError("division by zero")
    return base**exponent.numerator


def apply(operator: str, left: Fraction, right: Fraction) -> Fraction:
    match operator:
        case "+":
            return left + right
        case "-":
            return left - right
        case "*":
            return left * right
        case "/" | "%" if right == 0:
            raise CalculatorError("division by zero")
        case "/":
            return left / right
        case "%":
            return left % right
        case "^":
            return power(left, right)
        case _:
            raise CalculatorError(f"unknown operator '{operator}'")


@dataclass(slots=True)
class Evaluator:
    steps: list[str] = field(default_factory=list)

    def evaluate(self, node: Node) -> Fraction:
        match node:
            case Number(value):
                return value
            case Negate(operand):
                return -self.evaluate(operand)
            case Binary(operator, left, right):
                a = self.evaluate(left)
                b = self.evaluate(right)
                result = apply(operator, a, b)
                self.steps.append(f"{format_number(a)} {operator} {format_number(b)} = {format_number(result)}")
                return result
