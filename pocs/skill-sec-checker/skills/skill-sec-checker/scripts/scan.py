import hashlib
import os
import sys
from pathlib import Path

import rules
import store

SKIP = {".git", "__pycache__", ".DS_Store"}
MAX_SCAN_BYTES = 2 * 1024 * 1024
MAX_HITS_PER_RULE = 5
EXECUTABLE_MAGIC = (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"MZ")
INVISIBLE = set(range(0x200B, 0x2010)) | set(range(0x202A, 0x202F)) | set(range(0x2060, 0x2065)) | set(range(0x2066, 0x206A)) | {0xFEFF}


def default_roots():
    home = Path.home()
    return [("claude", home / ".claude" / "skills"), ("codex", home / ".codex" / "skills")]


def parse_roots(args):
    roots = default_roots()
    for arg in args:
        if "=" not in arg:
            sys.exit(f"root must be label=path, got {arg}")
        label, path = arg.split("=", 1)
        roots.append((label, Path(path).expanduser()))
    return roots


def discover(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = len(Path(dirpath).relative_to(root).parts)
        if "SKILL.md" in filenames and depth > 0:
            found.append(Path(dirpath))
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP) if depth < 2 else []
    return sorted(found)


def walk_skill(skill_dir):
    entries = []
    for dirpath, dirnames, filenames in os.walk(skill_dir):
        links = [d for d in dirnames if (Path(dirpath) / d).is_symlink()]
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP and d not in links)
        for name in sorted(filenames + links):
            if name not in SKIP:
                entries.append(Path(dirpath) / name)
    return entries


def visible(text):
    return "".join(f"<U+{ord(c):04X}>" if ord(c) in INVISIBLE else c for c in text)


def finding(rule, rel, line, excerpt):
    return {
        "id": f"{rule['id']}@{rel}:{line}",
        "rule": rule["id"],
        "severity": rule["severity"],
        "title": rule["title"],
        "file": rel,
        "line": line,
        "excerpt": visible(excerpt.strip())[:220],
    }


def scan_text(rel, text, hits):
    out = []
    name = Path(rel).name
    for number, line in enumerate(text.splitlines(), 1):
        for rule, regex in rules.COMPILED:
            if rule.get("only") and rule["only"] != name:
                continue
            if hits.get(rule["id"], 0) >= MAX_HITS_PER_RULE:
                continue
            if regex.search(line):
                hits[rule["id"]] = hits.get(rule["id"], 0) + 1
                out.append(finding(rule, rel, number, line))
    return out


def inspect(skill_dir, path, hits):
    rel = path.relative_to(skill_dir).as_posix()
    if path.is_symlink():
        target = os.readlink(path)
        entry = {"path": rel, "sha256": hashlib.sha256(f"symlink:{target}".encode()).hexdigest(), "size": 0, "kind": "symlink"}
        resolved = path.resolve()
        if skill_dir.resolve() not in resolved.parents:
            return entry, [finding(rules.ALL_RULES["symlink-escape"], rel, 0, f"-> {target}")]
        return entry, []
    data = path.read_bytes()
    entry = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "kind": "text"}
    if data.startswith(EXECUTABLE_MAGIC):
        entry["kind"] = "binary"
        return entry, [finding(rules.ALL_RULES["executable-binary"], rel, 0, "compiled executable header")]
    if b"\x00" in data[:8192]:
        entry["kind"] = "binary"
        return entry, [finding(rules.ALL_RULES["binary-file"], rel, 0, f"{len(data)} bytes")]
    if len(data) > MAX_SCAN_BYTES:
        entry["kind"] = "large"
        return entry, [finding(rules.ALL_RULES["oversized-file"], rel, 0, f"{len(data)} bytes")]
    return entry, scan_text(rel, data.decode("utf-8", errors="replace"), hits)


def skill_hash(files):
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda x: x["path"]):
        digest.update(f"{f['path']}\0{f['sha256']}\n".encode())
    return digest.hexdigest()


def diff_files(previous, files):
    current = {f["path"]: f["sha256"] for f in files}
    return {
        "added": sorted(p for p in current if p not in previous),
        "removed": sorted(p for p in previous if p not in current),
        "modified": sorted(p for p in current if p in previous and previous[p] != current[p]),
    }


def last_present(timeline):
    for entry in reversed(timeline):
        if not entry.get("removed"):
            return entry
    return None


def scan_skill(label, root, skill_dir, history):
    hits = {}
    files, findings = [], []
    for path in walk_skill(skill_dir):
        entry, found = inspect(skill_dir, path, hits)
        files.append(entry)
        findings.extend(found)
    skill_id = f"{label}/{skill_dir.relative_to(root).as_posix()}"
    digest = skill_hash(files)
    timeline = history["skills"].get(skill_id, {}).get("timeline", [])
    previous = last_present(timeline)
    if previous is None:
        status, changes = "new", None
    elif previous["hash"] == digest and not timeline[-1].get("removed"):
        status, changes = "unchanged", None
    else:
        status, changes = "changed", diff_files(previous.get("files", {}), files)
    return {
        "id": skill_id,
        "agent": label,
        "name": skill_dir.relative_to(root).as_posix(),
        "dir": str(skill_dir),
        "path": store.display(skill_dir),
        "hash": digest,
        "previousHash": previous["hash"] if previous else None,
        "status": status,
        "changes": changes,
        "files": files,
        "findings": findings,
    }


def run_dir(at):
    base = store.data_dir() / "runs" / at.replace(":", "").replace("-", "")
    candidate, n = base, 1
    while candidate.exists():
        n += 1
        candidate = base.with_name(f"{base.name}-{n}")
    candidate.mkdir(parents=True)
    return candidate


def main(args):
    at = store.now()
    roots = parse_roots(args)
    history = store.load_history()
    skills = []
    for label, root in roots:
        if root.is_dir():
            skills.extend(scan_skill(label, root, d, history) for d in discover(root))
    ids = {s["id"] for s in skills}
    removed = sorted(
        sid for sid, record in history["skills"].items()
        if sid not in ids and record["timeline"] and not record["timeline"][-1].get("removed")
    )
    pending, seen = [], set()
    for s in skills:
        s["reviewCached"] = s["hash"] in history["reviews"]
        if not s["reviewCached"] and s["hash"] not in seen:
            seen.add(s["hash"])
            pending.append(s["id"])
    out = run_dir(at)
    store.write_json(out / "scan.json", {
        "at": at,
        "runDir": str(out),
        "roots": [{"agent": label, "path": store.display(root), "exists": root.is_dir()} for label, root in roots],
        "skills": skills,
        "removed": removed,
        "pending": pending,
    })
    print(f"run dir: {out}")
    print(f"skills: {len(skills)}  new: {sum(s['status'] == 'new' for s in skills)}  changed: {sum(s['status'] == 'changed' for s in skills)}  unchanged: {sum(s['status'] == 'unchanged' for s in skills)}  removed: {len(removed)}")
    print(f"pending review: {len(pending)}")
    for sid in pending:
        skill = next(s for s in skills if s["id"] == sid)
        print(f"  {sid}  findings={len(skill['findings'])}  dir={skill['dir']}")


if __name__ == "__main__":
    main(sys.argv[1:])
