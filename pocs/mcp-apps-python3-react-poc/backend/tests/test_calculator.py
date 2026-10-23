import pytest

from calculator_app.calculator import calculate
from calculator_app.tokens import CalculatorError


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", "3"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("10 - 4 - 3", "3"),
        ("2 ^ 3 ^ 2", "512"),
        ("-2 ^ 2", "-4"),
        ("(-2) ^ 2", "4"),
        ("7 % 3", "1"),
        ("8 ÷ 2 × 3", "12"),
        ("5 − 8", "-3"),
        ("0.1 + 0.2", "0.3"),
        ("1 / 3", "0.333333333333"),
        ("2 ^ -2", "0.25"),
        ("--3", "3"),
    ],
)
def test_calculate_follows_arithmetic_rules(expression: str, expected: str):
    assert calculate(expression)["result"] == expected


def test_decimal_input_is_exact_so_money_style_sums_do_not_drift():
    assert calculate("0.1 + 0.1 + 0.1 - 0.3")["result"] == "0"


def test_steps_show_evaluation_order_respecting_precedence():
    assert calculate("2 + 3 * 4")["steps"] == ["3 * 4 = 12", "2 + 12 = 14"]


def test_expression_is_echoed_trimmed_for_the_display():
    assert calculate("  1+1 ")["expression"] == "1+1"


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("", "expression is empty"),
        ("1 / 0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("0 ^ -1", "division by zero"),
        ("2 ^ 0.5", "exponent must be a whole number"),
        ("2 ^ 5000", "exponent must be between"),
        ("1 +", "unexpected end of expression"),
        ("(1 + 2", "unexpected end of expression"),
        ("1 + 2)", "unexpected '\\)'"),
        ("1..2", "invalid number"),
        ("2 & 3", "unexpected character '&'"),
        ("import os", "unexpected character 'i'"),
    ],
)
def test_invalid_input_is_rejected_with_a_clear_reason(expression: str, message: str):
    with pytest.raises(CalculatorError, match=message):
        calculate(expression)
