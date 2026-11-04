import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from agentinfra import InfraError, call, log

CLIENT = "answer-grader"
HISTORY_LIMIT = 20


def load_checks(raw: str) -> list[dict[str, Any]]:
    path = Path(raw)
    checks = json.loads(path.read_text() if path.exists() else raw)
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must be a non empty JSON list")
    return checks


def history(suite: str) -> dict[str, Any]:
    try:
        return call(CLIENT, "state", "get", f"history/{suite}")["value"]
    except InfraError as error:
        if "not_found" in str(error):
            return {"suite": suite, "best": None, "scores": []}
        raise


def trend(scores: list[float], score: float) -> str:
    if not scores:
        return "first"
    if score > scores[-1]:
        return "better"
    if score < scores[-1]:
        return "worse"
    return "same"


def grade(suite: str, answer_file: Path, checks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    answer = answer_file.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        suite_file = Path(tmp) / "suite.json"
        suite_file.write_text(
            json.dumps({"suite": suite, "cases": [{"name": name, "input": answer_file.name, "output": answer, "checks": checks}]})
        )
        run = call(CLIENT, "eval", "run", str(suite_file))
    past = history(suite)
    score = float(run["score"])
    direction = trend(past["scores"], score)
    best = score if past["best"] is None else max(float(past["best"]), score)
    record = {"suite": suite, "best": best, "scores": (past["scores"] + [score])[-HISTORY_LIMIT:], "last_run": run["id"]}
    call(CLIENT, "state", "put", f"history/{suite}", json.dumps(record))
    log(CLIENT, "warn" if direction == "worse" else "info", "answer graded", suite=suite, score=score, run_id=run["id"], trend=direction)
    failed = [c for case in run["cases"] for c in case["checks"] if not c["passed"]]
    return {"run_id": run["id"], "suite": suite, "score": score, "trend": direction, "best": best, "failed_checks": failed}


def main() -> int:
    parser = argparse.ArgumentParser(prog="grade", description="Grade an answer with the shared evaluators and track its score")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--suite", required=True)
    g.add_argument("--answer", required=True, type=Path)
    g.add_argument("--checks", required=True, help="JSON list or path to a JSON file")
    g.add_argument("--name", default="answer")
    sub.add_parser("history").add_argument("--suite", required=True)
    sub.add_parser("evaluators")
    args = parser.parse_args()
    try:
        match args.command:
            case "grade":
                result = grade(args.suite, args.answer, load_checks(args.checks), args.name)
            case "history":
                result = history(args.suite)
            case _:
                result = call(CLIENT, "eval", "evaluators")
    except (InfraError, ValueError, FileNotFoundError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
