from collections.abc import Callable
from fractions import Fraction

type Number = int | float
type Operation = Callable[[Fraction, Fraction], Fraction]


class CalculatorError(ValueError):
    pass


def divide(a: Fraction, b: Fraction) -> Fraction:
    if b == 0:
        raise CalculatorError("division by zero")
    return a / b


OPERATIONS: dict[str, Operation] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": divide,
}


def to_fraction(value: str | Number) -> Fraction:
    try:
        return Fraction(str(value))
    except ValueError, ZeroDivisionError:
        raise CalculatorError(f"not a number: {value!r}") from None


def apply(operator: str, a: Fraction, b: Fraction) -> Fraction:
    if operator not in OPERATIONS:
        raise CalculatorError(f"unknown operator: {operator!r}")
    return OPERATIONS[operator](a, b)


def to_number(value: Fraction) -> Number:
    if value.denominator == 1:
        return value.numerator
    return float(value)
