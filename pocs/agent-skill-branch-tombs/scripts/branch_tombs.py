import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

SEP = "\x1f"


def run_git(args):
    return subprocess.run(["git"] + args, capture_output=True, text=True, errors="replace")


def repo_root():
    result = run_git(["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def detect_default():
    result = run_git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"])
    if result.returncode == 0 and result.stdout.strip():
        ref = result.stdout.strip()
        name = ref.split("/", 1)[1] if "/" in ref else ref
        return ref, name
    for candidate in ("main", "master"):
        if run_git(["rev-parse", "--verify", "--quiet", candidate]).returncode == 0:
            return candidate, candidate
    current = run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip() or "HEAD"
    return current, current


def remote_names():
    return set(p for p in run_git(["remote"]).stdout.split() if p)


def list_refs(remotes):
    fields = [
        "%(refname)",
        "%(refname:short)",
        "%(objectname:short)",
        "%(committerdate:unix)",
        "%(authorname)",
        "%(committername)",
        "%(contents:subject)",
    ]
    result = run_git(["for-each-ref", "--format=" + SEP.join(fields), "refs/heads", "refs/remotes"])
    refs = []
    for line in result.stdout.split("\n"):
        if not line.strip():
            continue
        parts = line.split(SEP)
        if len(parts) < 7:
            continue
        refname, short, obj, cdate, author, committer, subject = parts[:7]
        if refname.endswith("/HEAD"):
            continue
        try:
            when = int(cdate)
        except ValueError:
            when = 0
        if refname.startswith("refs/remotes/"):
            location = "remote"
            head = short.split("/", 1)
            if len(head) == 2 and head[0] in remotes:
                name, remote = head[1], head[0]
            else:
                name, remote = short, ""
        else:
            location = "local"
            name, remote = short, ""
        refs.append({
            "short": short,
            "name": name,
            "obj": obj,
            "when": when,
            "author": author.strip() or "Unknown",
            "committer": committer.strip() or "Unknown",
            "subject": subject.strip(),
            "location": location,
            "remote": remote,
        })
    return refs


def ahead_behind(default_ref, ref):
    result = run_git(["rev-list", "--left-right", "--count", default_ref + "..." + ref])
    if result.returncode != 0:
        return 0, 0
    parts = result.stdout.split()
    if len(parts) != 2:
        return 0, 0
    try:
        return int(parts[1]), int(parts[0])
    except ValueError:
        return 0, 0


def is_merged(default_ref, ref):
    return run_git(["merge-base", "--is-ancestor", ref, default_ref]).returncode == 0


def tier_of(age_days):
    if age_days <= 30:
        return "active"
    if age_days <= 90:
        return "stale"
    if age_days <= 365:
        return "abandoned"
    return "ancient"


def collapse(refs):
    grouped = defaultdict(lambda: {
        "name": None, "local": False, "remotes": set(), "best": None,
    })
    for ref in refs:
        entry = grouped[ref["name"]]
        entry["name"] = ref["name"]
        if ref["location"] == "local":
            entry["local"] = True
        elif ref["remote"]:
            entry["remotes"].add(ref["remote"])
        if entry["best"] is None or ref["when"] > entry["best"]["when"]:
            entry["best"] = ref
    return list(grouped.values())


def location_label(entry):
    if entry["local"] and entry["remotes"]:
        return "local+remote"
    if entry["local"]:
        return "local"
    if entry["remotes"]:
        return "remote"
    return "unknown"


def prune_commands(entry):
    commands = []
    if entry["local"]:
        commands.append("git branch -d " + entry["name"])
    for remote in sorted(entry["remotes"]):
        commands.append("git push " + remote + " --delete " + entry["name"])
    return commands


def analyze(default_ref, default_name, threshold, now_ts):
    remotes = remote_names()
    refs = list_refs(remotes)
    branches = []
    for entry in collapse(refs):
        best = entry["best"]
        name = entry["name"]
        is_default = name == default_name
        when = best["when"]
        age_days = max(0, int((now_ts - when) / 86400)) if when else 0
        ahead, behind = ahead_behind(default_ref, best["short"])
        merged = is_merged(default_ref, best["short"])
        last_date = (
            datetime.fromtimestamp(when, tz=timezone.utc).strftime("%Y-%m-%d")
            if when else "unknown"
        )
        tier = "trunk" if is_default else tier_of(age_days)
        stale = (not is_default) and age_days > threshold
        branches.append({
            "name": name,
            "author": best["author"],
            "subject": best["subject"],
            "hash": best["obj"],
            "last_date": last_date,
            "age_days": age_days,
            "ahead": ahead,
            "behind": behind,
            "merged": merged,
            "location": location_label(entry),
            "is_default": is_default,
            "tier": tier,
            "stale": stale,
            "safe": stale and merged,
            "unmerged_work": stale and not merged and ahead > 0,
            "prune": prune_commands(entry) if (stale and merged) else [],
        })

    branches.sort(key=lambda item: (item["is_default"], -item["age_days"]))
    living = [b for b in branches if not b["is_default"]]
    stale_branches = [b for b in living if b["stale"]]
    safe = [b for b in stale_branches if b["safe"]]
    risky = [b for b in stale_branches if b["unmerged_work"]]

    grave_counts = defaultdict(int)
    for branch in stale_branches:
        grave_counts[branch["author"]] += 1
    gravekeeper = max(grave_counts.items(), key=lambda kv: kv[1], default=("none", 0))

    oldest = max(living, key=lambda item: item["age_days"], default=None)
    authors = sorted({b["author"] for b in living})

    summary = {
        "branches": len(living),
        "stale": len(stale_branches),
        "safe": len(safe),
        "risky": len(risky),
        "authors": len(authors),
        "threshold": threshold,
        "default": default_name,
        "oldest": {
            "name": oldest["name"] if oldest else "none",
            "age_days": oldest["age_days"] if oldest else 0,
            "author": oldest["author"] if oldest else "none",
        },
        "gravekeeper": {"name": gravekeeper[0], "graves": gravekeeper[1]},
    }

    return {
        "summary": summary,
        "branches": branches,
        "graves": stale_branches,
        "prune": safe,
    }


def render(template_path, data, out_html):
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    payload = json.dumps(data).replace("</", "<\\/")
    html = template.replace("__GRAVEYARD_DATA__", payload)
    with open(out_html, "w", encoding="utf-8") as handle:
        handle.write(html)


def print_summary(data, out_html):
    summary = data["summary"]
    print("Branch Graveyard")
    print("  default branch     : %s" % summary["default"])
    print("  branches scanned   : %d" % summary["branches"])
    print("  stale (>%d days)    : %d" % (summary["threshold"], summary["stale"]))
    print("  safe to prune      : %d (merged + stale)" % summary["safe"])
    print("  unmerged graves    : %d (have unique work)" % summary["risky"])
    if summary["oldest"]["name"] != "none":
        print("  oldest branch      : %s (%d days, %s)" % (
            summary["oldest"]["name"], summary["oldest"]["age_days"], summary["oldest"]["author"]))
    if summary["gravekeeper"]["graves"]:
        print("  gravekeeper        : %s (%d stale branches)" % (
            summary["gravekeeper"]["name"], summary["gravekeeper"]["graves"]))
    print("")
    print("Oldest graves:")
    for branch in data["graves"][:6]:
        flag = "merged" if branch["merged"] else "UNMERGED +%d" % branch["ahead"]
        print("  [%4dd] %-28s %-18s %s" % (
            branch["age_days"], branch["name"][:28], branch["author"][:18], flag))
    print("")
    print("Report written to %s" % out_html)


def main(argv):
    threshold = 30
    for arg in argv[1:]:
        if arg.startswith("--days="):
            try:
                threshold = int(arg.split("=", 1)[1])
            except ValueError:
                pass
        elif arg.isdigit():
            threshold = int(arg)

    root = repo_root()
    if root is None:
        sys.stderr.write("not a git repository\n")
        return 2

    default_ref, default_name = detect_default()
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    data = analyze(default_ref, default_name, threshold, now_ts)
    data["repo"] = os.path.basename(root)
    data["generated"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out_dir = os.path.join(os.getcwd(), "branch-graveyard-report")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "data.json")
    out_html = os.path.join(out_dir, "index.html")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    here = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.normpath(os.path.join(here, "..", "assets", "template.html"))
    if os.path.exists(template_path):
        render(template_path, data, out_html)
    else:
        sys.stderr.write("template not found at %s\n" % template_path)

    print_summary(data, out_html)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
