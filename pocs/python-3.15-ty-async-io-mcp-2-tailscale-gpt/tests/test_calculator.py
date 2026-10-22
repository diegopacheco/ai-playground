from fractions import Fraction

import pytest

from calculator.arithmetic import CalculatorError, to_number
from calculator.english import evaluate_english, to_tokens
from calculator.infix import evaluate_infix
from calculator.polish import evaluate_polish


def test_infix_respects_precedence_so_multiplication_binds_tighter():
    assert evaluate_infix("1 + 2 * 3") == 7
    assert evaluate_infix("(1 + 2) * 3") == 9


def test_infix_is_left_associative_so_subtraction_chains_correctly():
    assert evaluate_infix("10 - 4 - 3") == 3
    assert evaluate_infix("100 / 10 / 5") == 2


def test_infix_supports_unary_minus():
    assert evaluate_infix("-3 * -(2 + 1)") == 9


def test_decimals_are_exact_so_users_do_not_see_float_noise():
    assert to_number(evaluate_infix("0.1 + 0.2")) == 0.3


def test_division_keeps_fractions_until_the_end():
    assert evaluate_infix("1 / 3 * 3") == 1


@pytest.mark.parametrize("expression", ["1 / 0", "/ 1 0", "one divided by zero"])
def test_division_by_zero_is_a_calculator_error(expression: str):
    evaluator = {"1": evaluate_infix, "/": evaluate_polish, "o": evaluate_english}[expression[0]]
    with pytest.raises(CalculatorError, match="division by zero"):
        evaluator(expression)


@pytest.mark.parametrize("expression", ["", "1 +", "(1 + 2", "1 2", "2 ^ 3", "* 2"])
def test_invalid_infix_is_rejected(expression: str):
    with pytest.raises(CalculatorError):
        evaluate_infix(expression)


def test_polish_operand_order_is_left_then_right():
    assert evaluate_polish("- 10 3") == 7
    assert evaluate_polish("/ 12 4") == 3


def test_polish_nests_operators():
    assert evaluate_polish("* + 1 2 3") == 9
    assert evaluate_polish("- * 2 5 / 8 4") == 8


@pytest.mark.parametrize("expression", ["", "+ 1", "1 2", "+ 1 2 3", "+ a 2"])
def test_invalid_polish_is_rejected(expression: str):
    with pytest.raises(CalculatorError):
        evaluate_polish(expression)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one plus two", 3),
        ("ten minus four", 6),
        ("six times seven", 42),
        ("ten divided by four", Fraction(5, 2)),
        ("what is twenty-one multiplied by two?", 42),
        ("one hundred and five plus five", 110),
        ("two thousand three hundred minus one", 2299),
        ("negative five plus two", -3),
        ("two plus three times four", 14),
        ("three plus 4", 7),
    ],
)
def test_english_phrases(text: str, expected: Fraction):
    assert evaluate_english(text) == expected


def test_english_compound_numbers_become_one_token():
    assert to_tokens("twenty one plus one hundred") == ["21", "+", "100"]


@pytest.mark.parametrize("text", ["", "one plus", "banana plus two", "one two"])
def test_invalid_english_is_rejected(text: str):
    with pytest.raises(CalculatorError):
        evaluate_english(text)
