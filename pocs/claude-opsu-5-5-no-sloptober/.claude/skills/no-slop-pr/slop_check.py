#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RULES = json.loads((Path(__file__).parent / "rules.json").read_text(encoding="utf-8"))
COMMENT = re.compile(r"^\s*(#|//|/\*|\*|--|<!--)")


def added_lines(diff):
    return [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]


def comment_ratio(diff):
    lines = added_lines(diff)
    if len(lines) < 5:
        return 0.0
    return sum(1 for line in lines if COMMENT.match(line)) / len(lines)


def verdict_for(score):
    if score >= RULES["block_at"]:
        return "BLOCK"
    if score >= RULES["suspect_at"]:
        return "SUSPECT"
    return "PASS"


def score(text, diff=""):
    hits = []
    for rule in RULES["rules"]:
        count = len(re.findall(rule["pattern"], text, re.IGNORECASE))
        if count:
            points = min(count * rule["weight"], rule["cap"])
            hits.append({"id": rule["id"], "label": rule["label"], "count": count, "points": points})
    ratio = comment_ratio(diff)
    for rule in RULES["diff_rules"]:
        if ratio >= rule["min_ratio"]:
            hits.append({"id": rule["id"], "label": rule["label"], "count": round(ratio, 2), "points": rule["weight"]})
    total = min(sum(hit["points"] for hit in hits), 100)
    return {"score": total, "verdict": verdict_for(total), "hits": hits}


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def fetch_pr(number, repo):
    extra = ["--repo", repo] if repo else []
    data = json.loads(gh("pr", "view", number, "--json", "title,body,commits", *extra))
    commits = "\n".join(
        c.get("messageHeadline", "") + "\n" + c.get("messageBody", "") for c in data.get("commits", [])
    )
    text = "\n".join([data.get("title", ""), data.get("body", ""), commits])
    return text, gh("pr", "diff", number, *extra)


def report(result):
    lines = [f"verdict: {result['verdict']}  score: {result['score']}/100"]
    for hit in result["hits"]:
        lines.append(f"  +{hit['points']:>3}  {hit['label']} ({hit['count']})")
    if not result["hits"]:
        lines.append("  no slop signals found")
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(description="Score a pull request for AI slop.")
    parser.add_argument("pr", nargs="?", help="pull request number or url, read with gh")
    parser.add_argument("--repo", help="owner/name when not inside the repo")
    parser.add_argument("--text", help="file with the PR title, body and commit messages")
    parser.add_argument("--diff", help="file with the unified diff")
    parser.add_argument("--json", action="store_true", help="print json")
    args = parser.parse_args(argv)
    if args.pr:
        text, diff = fetch_pr(args.pr, args.repo)
    elif args.text:
        text = Path(args.text).read_text(encoding="utf-8")
        diff = Path(args.diff).read_text(encoding="utf-8") if args.diff else ""
    else:
        parser.error("pass a PR number or --text")
    result = score(text, diff)
    print(json.dumps(result, indent=2) if args.json else report(result))
    return 1 if result["verdict"] == "BLOCK" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
