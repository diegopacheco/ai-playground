import json
import sys
from pathlib import Path

import rules
import store

SEVERITIES = tuple(rules.WEIGHTS)
TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "report.html"


def line_count(skill, rel):
    path = Path(skill["dir"]) / rel
    if not path.is_file():
        return None
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def validate(scan, reviews, history):
    errors = []
    skills = {s["id"]: s for s in scan["skills"]}
    reviewed = set()
    for review in reviews:
        sid = review.get("skill")
        skill = skills.get(sid)
        if skill is None:
            errors.append(f"review for '{sid}': no such skill in this scan")
            continue
        if sid in reviewed:
            errors.append(f"review for '{sid}': reviewed twice")
        reviewed.add(sid)
        summary = str(review.get("summary", "")).strip()
        if not summary:
            errors.append(f"{sid}: summary is empty")
        if len(summary) > 600:
            errors.append(f"{sid}: summary is {len(summary)} chars, the cap is 600")
        ids = {f["id"] for f in skill["findings"]}
        for d in review.get("dismiss", []):
            if d.get("finding") not in ids:
                errors.append(f"{sid}: dismissed finding '{d.get('finding')}' does not exist")
            if not str(d.get("reason", "")).strip():
                errors.append(f"{sid}: dismissal of '{d.get('finding')}' has no reason")
        known = {f["path"] for f in skill["files"]}
        for a in review.get("add", []):
            if a.get("severity") not in SEVERITIES:
                errors.append(f"{sid}: added finding severity must be one of {', '.join(SEVERITIES)}")
            if a.get("file") not in known:
                errors.append(f"{sid}: added finding file '{a.get('file')}' does not exist in the skill")
                continue
            total = line_count(skill, a["file"])
            if total is None or not isinstance(a.get("line"), int) or not 1 <= a["line"] <= total:
                errors.append(f"{sid}: added finding line {a.get('line')} is outside {a['file']} (1-{total})")
            for key in ("title", "why"):
                if not str(a.get(key, "")).strip():
                    errors.append(f"{sid}: added finding has no {key}")
    cached = set(history["reviews"])
    new_hashes = {skills[r["skill"]]["hash"] for r in reviews if r.get("skill") in skills}
    for sid in scan["pending"]:
        if skills[sid]["hash"] not in new_hashes and skills[sid]["hash"] not in cached:
            errors.append(f"{sid}: pending review is missing")
    return errors


def apply_review(skill, review, at):
    dismissed = {d["finding"]: d["reason"] for d in review.get("dismiss", [])}
    active, gone = [], []
    for f in skill["findings"]:
        if f["id"] in dismissed:
            gone.append(dict(f, reason=dismissed[f["id"]]))
        else:
            active.append(f)
    for n, a in enumerate(review.get("add", []), 1):
        active.append({
            "id": f"reviewer-{n}@{a['file']}:{a['line']}",
            "rule": f"reviewer-{n}",
            "severity": a["severity"],
            "title": a["title"],
            "why": a["why"],
            "file": a["file"],
            "line": a["line"],
            "excerpt": "",
            "byReviewer": True,
        })
    for f in active:
        if not f.get("byReviewer"):
            f["why"] = rules.ALL_RULES[f["rule"]]["why"]
    order = {s: i for i, s in enumerate(SEVERITIES)}
    active.sort(key=lambda f: (order[f["severity"]], f["file"], f["line"]))
    value = rules.score(active)
    key, label = rules.band(value)
    return {
        "score": value,
        "band": key,
        "bandLabel": label,
        "summary": review["summary"],
        "reviewedAt": review.get("reviewedAt", at),
        "findings": active,
        "dismissed": gone,
    }


def record(history, skill, result, at):
    entry = history["skills"].setdefault(skill["id"], {"timeline": []})
    last = entry["timeline"][-1] if entry["timeline"] else None
    if last and not last.get("removed") and last["hash"] == skill["hash"] and last["score"] == result["score"]:
        return
    entry["timeline"].append({
        "at": at,
        "hash": skill["hash"],
        "score": result["score"],
        "files": {f["path"]: f["sha256"] for f in skill["files"]},
    })


def main(run):
    run = Path(run)
    scan = store.read_json(run / "scan.json")
    history = store.load_history()
    review_file = run / "review.json"
    reviews = store.read_json(review_file)["reviews"] if review_file.exists() else []
    errors = validate(scan, reviews, history)
    if errors:
        print("review.json rejected:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    at = scan["at"]
    by_id = {s["id"]: s for s in scan["skills"]}
    for review in reviews:
        history["reviews"][by_id[review["skill"]]["hash"]] = {
            "summary": review["summary"].strip(),
            "dismiss": review.get("dismiss", []),
            "add": review.get("add", []),
            "reviewedAt": at,
        }
    out = []
    for skill in scan["skills"]:
        result = apply_review(skill, history["reviews"][skill["hash"]], at)
        record(history, skill, result, at)
        timeline = [{k: v for k, v in e.items() if k != "files"} for e in history["skills"][skill["id"]]["timeline"]]
        out.append({k: v for k, v in skill.items() if k not in ("dir", "findings")} | result | {"timeline": timeline})
    removed = []
    for sid in scan["removed"]:
        timeline = history["skills"][sid]["timeline"]
        last = next(e for e in reversed(timeline) if not e.get("removed"))
        removed.append({"id": sid, "lastHash": last["hash"], "lastScore": last["score"], "lastSeen": last["at"]})
        if not timeline[-1].get("removed"):
            timeline.append({"at": at, "removed": True})
    scores = [s["score"] for s in out]
    summary = {
        "at": at,
        "dir": store.display(run),
        "skills": len(out),
        "average": round(sum(scores) / len(scores), 1) if scores else None,
        "dangerous": sum(s["band"] == "dangerous" for s in out),
        "new": sum(s["status"] == "new" for s in out),
        "changed": sum(s["status"] == "changed" for s in out),
        "unchanged": sum(s["status"] == "unchanged" for s in out),
        "removed": len(removed),
        "reviewed": len(reviews),
    }
    history["runs"] = [r for r in history["runs"] if r["dir"] != summary["dir"]] + [summary]
    store.save_history(history)
    payload = {
        "at": at,
        "runDir": store.display(run),
        "dataDir": store.display(store.data_dir()),
        "roots": scan["roots"],
        "summary": summary,
        "skills": out,
        "removed": removed,
        "runs": history["runs"],
        "rules": rules.public_rules(),
        "bands": [{"floor": f, "key": k, "label": l} for f, k, l in rules.BANDS],
    }
    data = json.dumps(payload).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__PAYLOAD__", data)
    (run / "index.html").write_text(html, encoding="utf-8")
    print(f"report: {run / 'index.html'}")
    print(f"skills: {summary['skills']}  average: {summary['average']}  dangerous: {summary['dangerous']}  removed: {summary['removed']}")
    for s in sorted(out, key=lambda x: x["score"])[:5]:
        print(f"  {s['score']:>2}/10  {s['bandLabel']:<13} {s['id']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: render.py RUN_DIR")
    main(sys.argv[1])
