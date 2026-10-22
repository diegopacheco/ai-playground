from fractions import Fraction

from calculator.arithmetic import OPERATIONS, CalculatorError, apply, to_fraction


def evaluate_polish(expression: str) -> Fraction:
    tokens = expression.split()
    if not tokens:
        raise CalculatorError("empty expression")
    stack: list[Fraction] = []
    for token in reversed(tokens):
        if token in OPERATIONS:
            if len(stack) < 2:
                raise CalculatorError(f"operator {token!r} needs two operands")
            left = stack.pop()
            right = stack.pop()
            stack.append(apply(token, left, right))
        else:
            stack.append(to_fraction(token))
    if len(stack) != 1:
        raise CalculatorError("too many operands")
    return stack[0]
