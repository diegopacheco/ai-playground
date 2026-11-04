import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseInput:
    output: str
    latency_ms: int


@dataclass(frozen=True)
class CheckResult:
    type: str
    expected: Any
    passed: bool
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {"type": self.type, "expected": self.expected, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class Evaluator:
    name: str
    description: str
    value_hint: str
    grade: Callable[[CaseInput, Any], tuple[bool, str]]


def _exact_match(case: CaseInput, expected: Any) -> tuple[bool, str]:
    ok = case.output.strip() == str(expected).strip()
    return ok, "output equals expected" if ok else "output differs from expected"


def _contains(case: CaseInput, expected: Any) -> tuple[bool, str]:
    ok = str(expected).lower() in case.output.lower()
    return ok, f"{'found' if ok else 'missing'}: {expected}"


def _not_contains(case: CaseInput, expected: Any) -> tuple[bool, str]:
    ok = str(expected).lower() not in case.output.lower()
    return ok, f"{'absent' if ok else 'present'}: {expected}"


def _regex(case: CaseInput, expected: Any) -> tuple[bool, str]:
    try:
        ok = re.search(str(expected), case.output, re.MULTILINE) is not None
    except re.error as error:
        return False, f"bad pattern: {error}"
    return ok, f"pattern {'matched' if ok else 'not matched'}"


def _parse(output: str) -> tuple[Any, str | None]:
    try:
        return json.loads(output), None
    except ValueError as error:
        return None, str(error)


def _json_valid(case: CaseInput, _: Any) -> tuple[bool, str]:
    _, error = _parse(case.output)
    return error is None, "valid JSON" if error is None else f"invalid JSON: {error}"


def _lookup(document: Any, path: str) -> tuple[bool, Any]:
    current = document
    for part in [p for p in path.split(".") if p]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def strict_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(strict_equal(a, e) for a, e in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(strict_equal(actual[k], expected[k]) for k in actual)
    return actual == expected


def _json_field_equals(case: CaseInput, expected: Any) -> tuple[bool, str]:
    if not isinstance(expected, dict) or "path" not in expected or "equals" not in expected:
        return False, 'expected must be {"path": "...", "equals": ...}'
    document, error = _parse(case.output)
    if error is not None:
        return False, f"invalid JSON: {error}"
    found, actual = _lookup(document, str(expected["path"]))
    if not found:
        return False, f"path {expected['path']} not found"
    ok = strict_equal(actual, expected["equals"])
    return ok, f"{expected['path']} = {json.dumps(actual)}"


def _max_length(case: CaseInput, expected: Any) -> tuple[bool, str]:
    ok = len(case.output) <= int(expected)
    return ok, f"length {len(case.output)} <= {expected}" if ok else f"length {len(case.output)} > {expected}"


def _min_length(case: CaseInput, expected: Any) -> tuple[bool, str]:
    ok = len(case.output) >= int(expected)
    return ok, f"length {len(case.output)} >= {expected}" if ok else f"length {len(case.output)} < {expected}"


def _max_latency_ms(case: CaseInput, expected: Any) -> tuple[bool, str]:
    ok = case.latency_ms <= int(expected)
    return ok, f"{case.latency_ms}ms {'<=' if ok else '>'} {expected}ms"


EVALUATORS: dict[str, Evaluator] = {
    e.name: e
    for e in [
        Evaluator("exact_match", "Output equals the expected text, ignoring surrounding whitespace", "text", _exact_match),
        Evaluator("contains", "Output contains the text, case insensitive", "text", _contains),
        Evaluator("not_contains", "Output does not contain the text, case insensitive", "text", _not_contains),
        Evaluator("regex", "Output matches the regular expression", "pattern", _regex),
        Evaluator("json_valid", "Output parses as JSON", "none", _json_valid),
        Evaluator("json_field_equals", "JSON field at a dotted path equals a value", '{"path","equals"}', _json_field_equals),
        Evaluator("max_length", "Output has at most N characters", "int", _max_length),
        Evaluator("min_length", "Output has at least N characters", "int", _min_length),
        Evaluator("max_latency_ms", "Case finished within N milliseconds", "int", _max_latency_ms),
    ]
}


def grade(case: CaseInput, checks: list[dict[str, Any]]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        name = str(check.get("type", ""))
        expected = check.get("value")
        evaluator = EVALUATORS.get(name)
        if evaluator is None:
            results.append(CheckResult(name, expected, False, f"unknown evaluator {name}"))
            continue
        try:
            passed, detail = evaluator.grade(case, expected)
        except (TypeError, ValueError) as error:
            passed, detail = False, f"bad value: {error}"
        results.append(CheckResult(name, expected, passed, detail))
    return results
