import re
from collections.abc import Mapping
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

HEADER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
PARAM = re.compile(r"\$([a-z_]+)")


def render_prompt(template: str, params: Mapping[str, str]) -> str:
    match = HEADER.match(template)
    if not match:
        raise ValueError("prompt is missing its --- header")
    declared = PARAM.findall(match.group(1))
    missing = [name for name in declared if not params.get(name, "").strip()]
    if missing:
        raise ValueError(f"prompt parameters without value: {', '.join(f'${name}' for name in missing)}")
    body = match.group(2)
    for name in sorted(declared, key=len, reverse=True):
        body = body.replace(f"${name}", params[name])
    return body.strip()


def load_prompt(name: str, params: Mapping[str, str], directory: Path = PROMPTS_DIR) -> str:
    return render_prompt((directory / f"{name}.md").read_text(encoding="utf-8"), params)
