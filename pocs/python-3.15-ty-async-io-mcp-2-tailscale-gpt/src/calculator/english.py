import re
from fractions import Fraction

from calculator.arithmetic import CalculatorError
from calculator.infix import Parser

UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SMALL: dict[str, int] = {word: value for value, word in enumerate(UNITS)} | {
    word: (index + 2) * 10 for index, word in enumerate(TENS)
}
SCALES: dict[str, int] = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
OPERATORS: dict[str, str] = {
    "plus": "+", "add": "+",
    "minus": "-", "subtract": "-", "less": "-", "negative": "-",
    "times": "*", "multiplied": "*", "multiply": "*",
    "divided": "/", "divide": "/", "over": "/",
}
FILLER = {"by", "and", "what", "is", "the", "equals", "equal", "to", "of", "please"}


class NumberBuilder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.current = 0
        self.active = False

    def add(self, word: str) -> None:
        if word in SMALL:
            last = self.current % 100
            if last and (SMALL[word] >= 10 or last < 20 or last % 10):
                raise CalculatorError(f"unexpected number word: {word!r}")
            self.current += SMALL[word]
        elif word == "hundred":
            self.current = max(self.current, 1) * 100
        else:
            self.total += max(self.current, 1) * SCALES[word]
            self.current = 0
        self.active = True

    def flush(self) -> list[str]:
        if not self.active:
            return []
        value = self.total + self.current
        self.reset()
        return [str(value)]


def to_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    number = NumberBuilder()
    for word in re.findall(r"\d+\.?\d*|[a-z]+|[^\sa-z]", re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower())):
        if word in SMALL or word in SCALES or word == "hundred":
            number.add(word)
            continue
        if word in FILLER or word in "?.!":
            continue
        tokens += number.flush()
        if word in OPERATORS:
            tokens.append(OPERATORS[word])
        elif word[0].isdigit() or word in "+-*/()":
            tokens.append(word)
        else:
            raise CalculatorError(f"unknown word: {word!r}")
    tokens += number.flush()
    if not tokens:
        raise CalculatorError("empty expression")
    return tokens


def evaluate_english(text: str) -> Fraction:
    return Parser(to_tokens(text)).parse()
