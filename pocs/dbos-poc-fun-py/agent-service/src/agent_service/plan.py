import json

MAX_TASKS = 4


def parse_plan(output: str) -> list[str]:
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"agent returned no JSON task list: {output[:200]}")
    value: object = json.loads(output[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("agent plan is not a list")
    tasks = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not tasks:
        raise ValueError("agent plan has no tasks")
    return tasks[:MAX_TASKS]
