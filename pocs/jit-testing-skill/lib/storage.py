from pathlib import Path
import json
import time

def run_id():
    return time.strftime("%Y%m%dT%H%M%S")

def run_dir(repo: Path, rid: str) -> Path:
    return Path(repo) / ".jit-testing" / "runs" / rid

def ensure_gitignore(repo: Path):
    base = Path(repo) / ".jit-testing"
    base.mkdir(parents=True, exist_ok=True)
    gi = base / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n")

def write_catches(rdir: Path, meta: dict):
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "catches.json").write_text(json.dumps(meta, indent=2))

def load_catches(rdir: Path) -> dict:
    f = rdir / "catches.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())

def all_runs(repo: Path):
    base = Path(repo) / ".jit-testing" / "runs"
    if not base.exists():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir()])

def update_verdict(rdir: Path, catch_id: str, verdict: str):
    meta = load_catches(rdir)
    for c in meta.get("catches", []):
        if c.get("id") == catch_id:
            c["verdict"] = verdict
            c["verdict_ts"] = int(time.time())
            break
    write_catches(rdir, meta)
