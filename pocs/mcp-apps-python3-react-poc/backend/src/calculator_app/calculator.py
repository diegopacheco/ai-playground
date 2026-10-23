from typing import TypedDict

from calculator_app.evaluator import Evaluator, format_number
from calculator_app.parser import parse


class Calculation(TypedDict):
    expression: str
    result: str
    steps: list[str]


def calculate(expression: str) -> Calculation:
    evaluator = Evaluator()
    value = evaluator.evaluate(parse(expression))
    return Calculation(expression=expression.strip(), result=format_number(value), steps=evaluator.steps)
