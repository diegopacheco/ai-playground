import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    fixed = os.environ.get("SKILL_SEC_CHECKER_NOW")
    if fixed:
        return fixed
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def data_dir():
    custom = os.environ.get("SKILL_SEC_CHECKER_HOME")
    return Path(custom) if custom else Path.home() / ".skill-sec-checker"


def display(path):
    text = str(path)
    home = str(Path.home())
    if text == home or text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


def load_history():
    path = data_dir() / "history.json"
    if not path.exists():
        return {"version": 1, "runs": [], "skills": {}, "reviews": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_history(history):
    path = data_dir() / "history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
