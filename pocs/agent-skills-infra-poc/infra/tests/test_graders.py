import unittest

from evals.graders import EVALUATORS, CaseInput, grade
from evals.store import grade_cases
from common.http import HttpError


def one(output: str, check_type: str, value: object = None, latency: int = 0) -> bool:
    return grade(CaseInput(output, latency), [{"type": check_type, "value": value}])[0].passed


class GradersTest(unittest.TestCase):
    def test_catalog_lists_every_common_evaluator(self) -> None:
        self.assertEqual(
            set(EVALUATORS),
            {"exact_match", "contains", "not_contains", "regex", "json_valid", "json_field_equals", "max_length", "min_length", "max_latency_ms"},
        )

    def test_exact_match_ignores_surrounding_whitespace_only(self) -> None:
        self.assertTrue(one("  same\n", "exact_match", "same"))
        self.assertFalse(one("Same", "exact_match", "same"))

    def test_contains_and_not_contains_are_case_insensitive(self) -> None:
        self.assertTrue(one("Breaking Changes", "contains", "breaking changes"))
        self.assertFalse(one("has TODO", "not_contains", "todo"))

    def test_regex_matches_per_line_and_bad_pattern_fails_instead_of_crashing(self) -> None:
        self.assertTrue(one("intro\n# Release 1.2.3", "regex", r"^# Release \d+\.\d+\.\d+"))
        self.assertFalse(one("x", "regex", "("))

    def test_json_field_equals_walks_objects_and_lists(self) -> None:
        output = '{"items": [{"id": "a"}, {"id": "b"}], "count": 2}'
        self.assertTrue(one(output, "json_field_equals", {"path": "items.1.id", "equals": "b"}))
        self.assertFalse(one(output, "json_field_equals", {"path": "count", "equals": 3}))
        self.assertFalse(one(output, "json_field_equals", {"path": "missing", "equals": None}))
        self.assertFalse(one("not json", "json_field_equals", {"path": "a", "equals": 1}))

    def test_json_field_equals_distinguishes_one_from_true(self) -> None:
        self.assertFalse(one('{"on": 1}', "json_field_equals", {"path": "on", "equals": True}))
        self.assertFalse(one('{"on": [0]}', "json_field_equals", {"path": "on", "equals": [False]}))
        self.assertTrue(one('{"score": 1}', "json_field_equals", {"path": "score", "equals": 1.0}))

    def test_length_and_latency_bounds_are_inclusive(self) -> None:
        self.assertTrue(one("abc", "max_length", 3))
        self.assertFalse(one("abcd", "max_length", 3))
        self.assertTrue(one("abc", "min_length", 3))
        self.assertTrue(one("", "max_latency_ms", 100, latency=100))
        self.assertFalse(one("", "max_latency_ms", 100, latency=101))

    def test_unknown_evaluator_fails_the_check_so_typos_do_not_pass_silently(self) -> None:
        result = grade(CaseInput("x", 0), [{"type": "contain", "value": "x"}])[0]
        self.assertFalse(result.passed)
        self.assertIn("unknown evaluator", result.detail)

    def test_bad_value_type_fails_the_check(self) -> None:
        self.assertFalse(one("abc", "max_length", "ten"))

    def test_a_case_passes_only_when_every_check_passes(self) -> None:
        cases = grade_cases(
            [{"name": "c", "output": "hello", "checks": [{"type": "contains", "value": "hell"}, {"type": "max_length", "value": 2}]}]
        )
        self.assertFalse(cases[0].passed)

    def test_cases_without_checks_are_rejected_because_they_would_always_pass(self) -> None:
        with self.assertRaises(HttpError):
            grade_cases([{"name": "c", "output": "x", "checks": []}])
        with self.assertRaises(HttpError):
            grade_cases([])


if __name__ == "__main__":
    unittest.main()
