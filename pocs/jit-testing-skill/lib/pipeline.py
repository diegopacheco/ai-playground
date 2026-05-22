from pathlib import Path
import argparse
import subprocess
import time
import hashlib
from lib.detect import detect_target
from lib.runners import get_runner
from lib.rubfake import score_test
from lib.storage import write_catches, run_id, run_dir, ensure_gitignore

def run_pipeline(args):
    p = argparse.ArgumentParser(prog="jit run")
    p.add_argument("--repo", default=".")
    p.add_argument("--diff", default="HEAD~1..HEAD")
    p.add_argument("--target", default=None)
    p.add_argument("--workflow", default="both", choices=["dodgy", "intent", "both"])
    p.add_argument("--mode", default="auto", choices=["auto", "git", "snapshot"])
    p.add_argument("--parent-dir", default=".parent")
    p.add_argument("--max-tests", type=int, default=20)
    ns = p.parse_args(args)

    repo = Path(ns.repo).resolve()
    target = ns.target or detect_target(repo)
    if not target:
        print("no target detected; pass --target explicitly")
        return 1

    runner = get_runner(target)
    if runner is None:
        print(f"no runner registered for target: {target}")
        return 1

    mode = ns.mode
    if mode == "auto":
        mode = "snapshot" if (repo / ns.parent_dir).exists() else "git"
    runner.mode = mode
    runner.parent_dir = ns.parent_dir

    rid = run_id()
    ensure_gitignore(repo)
    rdir = run_dir(repo, rid)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "tests").mkdir(exist_ok=True)
    (rdir / "traces").mkdir(exist_ok=True)
    (rdir / "mutants").mkdir(exist_ok=True)

    if mode == "snapshot":
        intent = _read_intent_snapshot(repo, ns.parent_dir)
    else:
        intent = _read_intent(repo, ns.diff)

    catches = []
    if ns.workflow in ("dodgy", "both"):
        for c in runner.dodgy_diff(repo, ns.diff, rdir, ns.max_tests):
            c["workflow"] = "dodgy"
            catches.append(c)
    if ns.workflow in ("intent", "both"):
        for c in runner.intent_aware(repo, ns.diff, rdir, intent, ns.max_tests):
            c["workflow"] = "intent"
            catches.append(c)

    for c in catches:
        c.setdefault("id", _catch_id(c))
        c["intent_title"] = intent.get("title", "")
        c["rubfake"] = score_test(c)

    catches.sort(key=lambda c: -c["rubfake"]["score"])

    meta = {
        "run_id": rid,
        "target": target,
        "workflow": ns.workflow,
        "mode": mode,
        "diff": ns.diff if mode == "git" else f"snapshot:{ns.parent_dir}",
        "intent": intent,
        "started_at": int(time.time()),
        "catches": catches,
    }
    write_catches(rdir, meta)
    _write_report(rdir, meta)

    print(f"run_id: {rid}")
    print(f"target: {target}")
    print(f"mode: {mode}")
    print(f"catches: {len(catches)}")
    surfaced = [c for c in catches if c["rubfake"]["score"] > 0.3]
    print(f"surfaced: {len(surfaced)}")
    print(f"report: {rdir / 'report.md'}")
    return 0

def _catch_id(c: dict) -> str:
    key = (c.get("sense_check") or "") + (c.get("behavior_input") or "") + (c.get("name") or "")
    return hashlib.sha1(key.encode()).hexdigest()[:10]

def _read_intent_snapshot(repo: Path, parent_dir: str) -> dict:
    f = Path(repo) / parent_dir / "INTENT"
    if not f.exists():
        return {"title": "", "body": ""}
    text = f.read_text().strip()
    if "\n\n" in text:
        title, body = text.split("\n\n", 1)
    else:
        title, body = text.splitlines()[0] if text else "", ""
    return {"title": title.strip(), "body": body.strip()}

def _read_intent(repo: Path, diff: str) -> dict:
    title, body = "", ""
    try:
        ref = diff.split("..")[-1] if ".." in diff else diff
        title = subprocess.check_output(
            ["git", "-C", str(repo), "log", "-1", "--format=%s", ref],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        body = subprocess.check_output(
            ["git", "-C", str(repo), "log", "-1", "--format=%b", ref],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    return {"title": title, "body": body}

def _write_report(rdir: Path, meta: dict):
    out = ["# JIT Catching Run", ""]
    out.append(f"- run_id: `{meta['run_id']}`")
    out.append(f"- target: `{meta['target']}`")
    out.append(f"- workflow: `{meta['workflow']}`")
    out.append(f"- diff: `{meta['diff']}`")
    out.append(f"- intent title: {meta['intent'].get('title') or '(none)'}")
    out.append("")
    out.append(f"## Catches ({len(meta['catches'])})")
    out.append("")
    if not meta["catches"]:
        out.append("_No behavior differences detected._")
    for i, c in enumerate(meta["catches"], 1):
        sc = c["rubfake"]["score"]
        out.append(f"### {i}. score `{sc:+.2f}` — {c.get('sense_check') or c.get('name')}")
        out.append("")
        out.append(f"- id: `{c['id']}`")
        out.append(f"- workflow: `{c.get('workflow')}`")
        out.append(f"- fp patterns: {', '.join(c['rubfake']['fp_patterns']) or 'none'}")
        out.append(f"- tp patterns: {', '.join(c['rubfake']['tp_patterns']) or 'none'}")
        if c.get("behavior_input"):
            out.append(f"- input: `{c['behavior_input']}`")
        if c.get("parent_output") is not None or c.get("diff_output") is not None:
            out.append(f"- parent output: `{c.get('parent_output')}`")
            out.append(f"- diff output: `{c.get('diff_output')}`")
        if c.get("test_code"):
            out.append("")
            out.append("```")
            out.append(c["test_code"])
            out.append("```")
        out.append("")
    (rdir / "report.md").write_text("\n".join(out))
