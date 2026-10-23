import os
from pathlib import Path

VIEW_URI = "ui://calculator/view.html"
VIEW_FILE = "view.html"
HOST_FILE = "host.html"


def dist_dir() -> Path:
    configured = os.environ.get("CALCULATOR_DIST")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def read_page(dist: Path, name: str) -> str:
    page = dist / name
    if not page.is_file():
        raise FileNotFoundError(f"{page} is missing, build the frontend first")
    return page.read_text(encoding="utf-8")
