import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

DENY_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "target", "out",
    "bin", "obj", "__pycache__", ".next", ".nuxt", ".venv", "venv", "env",
    "coverage", ".gradle", ".idea", ".vscode", ".terraform", ".mypy_cache",
    ".pytest_cache", "Pods", "DerivedData", ".dart_tool", "tmp", ".cache",
}

DENY_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lock", "bun.lockb",
    "Cargo.lock", "go.sum", "poetry.lock", "composer.lock", "Gemfile.lock",
    "Pipfile.lock", "mix.lock", "flake.lock", ".DS_Store",
}

BINARY_EXT = (
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".tiff",
    ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".7z", ".rar", ".jar",
    ".war", ".class", ".o", ".a", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".mov", ".avi",
    ".wav", ".flac", ".webm", ".mkv", ".psd", ".sketch", ".ai", ".eps",
    ".lock", ".sum", ".min.js", ".min.css", ".map", ".snap", ".pyc", ".pdb",
)


def run_git(root, args):
    return subprocess.run(
        ["git", "-C", root] + args,
        capture_output=True, text=True, errors="replace",
    )


def repo_root(start):
    result = subprocess.run(
        ["git", "-C", start, "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def list_tracked(root, scope):
    args = ["ls-files", "-z"]
    if scope:
        args += ["--", scope]
    result = run_git(root, args)
    if result.returncode != 0:
        return []
    return [p for p in result.stdout.split("\0") if p]


def is_candidate(path):
    parts = path.split("/")
    for segment in parts[:-1]:
        if segment in DENY_DIRS:
            return False
    name = parts[-1]
    if name in DENY_FILES:
        return False
    lower = name.lower()
    if lower.endswith(BINARY_EXT):
        return False
    return True


def blame_file(root, path):
    result = run_git(root, ["blame", "--line-porcelain", "-w", "-M", "-C", "HEAD", "--", path])
    if result.returncode != 0 or not result.stdout:
        return None
    counts = defaultdict(int)
    author_recent = defaultdict(int)
    newest = 0
    current_author = None
    current_time = 0
    for line in result.stdout.split("\n"):
        if line.startswith("author "):
            current_author = line[7:].strip() or "Unknown"
        elif line.startswith("committer-time "):
            try:
                current_time = int(line[15:].strip())
            except ValueError:
                current_time = 0
        elif line.startswith("\t"):
            if current_author is None:
                continue
            counts[current_author] += 1
            if current_time > author_recent[current_author]:
                author_recent[current_author] = current_time
            if current_time > newest:
                newest = current_time
    if not counts:
        return None
    return counts, newest


def bus_factor(sorted_counts, total):
    cumulative = 0
    for index, value in enumerate(sorted_counts):
        cumulative += value
        if cumulative * 2 >= total:
            return index + 1
    return len(sorted_counts)


def tier_of(risk):
    if risk >= 75:
        return "critical"
    if risk >= 50:
        return "high"
    if risk >= 25:
        return "medium"
    return "low"


def file_risk(top_share, factor, idle_days):
    concentration_risk = top_share * 60.0
    if factor <= 1:
        spread_risk = 25.0
    elif factor == 2:
        spread_risk = 12.0
    else:
        spread_risk = 0.0
    idle_ratio = min(idle_days / 365.0, 2.0) / 2.0
    staleness_risk = idle_ratio * 15.0
    return round(min(concentration_risk + spread_risk + staleness_risk, 100.0), 1)


def grade_of(weighted_risk):
    if weighted_risk < 15:
        return "A"
    if weighted_risk < 30:
        return "B"
    if weighted_risk < 45:
        return "C"
    if weighted_risk < 60:
        return "D"
    return "F"


def top_segment(path):
    parts = path.split("/")
    if len(parts) == 1:
        return "(root)"
    return parts[0]


def analyze(root, scope, now_ts):
    tracked = [p for p in list_tracked(root, scope) if is_candidate(p)]
    files = []
    repo_author_lines = defaultdict(int)
    sole_files = defaultdict(int)
    sole_lines = defaultdict(int)
    touched_files = defaultdict(int)
    for path in tracked:
        blamed = blame_file(root, path)
        if blamed is None:
            continue
        counts, newest = blamed
        total = sum(counts.values())
        if total == 0:
            continue
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        sorted_values = [value for _, value in ranked]
        top_author, top_lines = ranked[0]
        top_share = top_lines / total
        factor = bus_factor(sorted_values, total)
        idle_days = max(0, int((now_ts - newest) / 86400)) if newest else 0
        risk = file_risk(top_share, factor, idle_days)
        for author, value in counts.items():
            repo_author_lines[author] += value
            touched_files[author] += 1
        single_owner = factor <= 1 and top_share >= 0.8
        if single_owner:
            sole_files[top_author] += 1
            sole_lines[top_author] += total
        last_date = (
            datetime.fromtimestamp(newest, tz=timezone.utc).strftime("%Y-%m-%d")
            if newest else "unknown"
        )
        files.append({
            "path": path,
            "dir": top_segment(path),
            "loc": total,
            "top_author": top_author,
            "top_share": round(top_share * 100, 1),
            "authors": len(counts),
            "bus_factor": factor,
            "idle_days": idle_days,
            "last_date": last_date,
            "single_owner": single_owner,
            "risk": risk,
            "tier": tier_of(risk),
        })

    total_loc = sum(item["loc"] for item in files)
    weighted_risk = (
        sum(item["risk"] * item["loc"] for item in files) / total_loc
        if total_loc else 0.0
    )
    repo_ranked = sorted(repo_author_lines.values(), reverse=True)
    repo_factor = bus_factor(repo_ranked, sum(repo_ranked)) if repo_ranked else 0

    authors = []
    for name, lines in sorted(repo_author_lines.items(), key=lambda item: item[1], reverse=True):
        authors.append({
            "name": name,
            "loc": lines,
            "share": round(lines / total_loc * 100, 1) if total_loc else 0.0,
            "sole_files": sole_files.get(name, 0),
            "sole_loc": sole_lines.get(name, 0),
            "files_touched": touched_files.get(name, 0),
        })

    dir_loc = defaultdict(int)
    dir_files = defaultdict(int)
    dir_risk_acc = defaultdict(float)
    dir_single = defaultdict(int)
    for item in files:
        key = item["dir"]
        dir_loc[key] += item["loc"]
        dir_files[key] += 1
        dir_risk_acc[key] += item["risk"] * item["loc"]
        if item["single_owner"]:
            dir_single[key] += 1
    directories = []
    for key in dir_loc:
        loc = dir_loc[key]
        directories.append({
            "path": key,
            "loc": loc,
            "files": dir_files[key],
            "risk": round(dir_risk_acc[key] / loc, 1) if loc else 0.0,
            "single_owner_files": dir_single[key],
        })
    directories.sort(key=lambda item: item["risk"] * item["loc"], reverse=True)

    single_files = [item for item in files if item["single_owner"]]
    single_loc = sum(item["loc"] for item in single_files)
    top_owner = authors[0] if authors else {"name": "none", "files_touched": 0, "loc": 0}
    risk_owner = max(
        authors, key=lambda item: item["sole_loc"], default=None
    ) if authors else None

    files.sort(key=lambda item: (item["risk"], item["loc"]), reverse=True)

    summary = {
        "files": len(files),
        "loc": total_loc,
        "authors": len(authors),
        "repo_bus_factor": repo_factor,
        "single_owner_files": len(single_files),
        "single_owner_loc": single_loc,
        "single_owner_pct": round(single_loc / total_loc * 100, 1) if total_loc else 0.0,
        "avg_risk": round(weighted_risk, 1),
        "grade": grade_of(weighted_risk),
        "top_owner": {
            "name": top_owner["name"],
            "loc": top_owner["loc"],
            "share": top_owner.get("share", 0.0),
        },
        "risk_owner": {
            "name": risk_owner["name"] if risk_owner else "none",
            "sole_files": risk_owner["sole_files"] if risk_owner else 0,
            "sole_loc": risk_owner["sole_loc"] if risk_owner else 0,
        },
    }

    return {
        "summary": summary,
        "authors": authors,
        "directories": directories,
        "files": files,
    }


def render(template_path, data, out_html):
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    payload = json.dumps(data).replace("</", "<\\/")
    html = template.replace("__BUS_FACTOR_DATA__", payload)
    with open(out_html, "w", encoding="utf-8") as handle:
        handle.write(html)


def print_summary(data, out_html):
    summary = data["summary"]
    print("Bus-Factor Report")
    print("  files analyzed     : %d (%d LOC)" % (summary["files"], summary["loc"]))
    print("  contributors       : %d" % summary["authors"])
    print("  repo bus factor    : %d" % summary["repo_bus_factor"])
    print("  single-owner files : %d (%.1f%% of code)" % (
        summary["single_owner_files"], summary["single_owner_pct"]))
    print("  knowledge grade    : %s (avg risk %.1f)" % (
        summary["grade"], summary["avg_risk"]))
    risk_owner = summary["risk_owner"]
    if risk_owner["sole_files"]:
        print("  biggest exposure   : %s solely owns %d files / %d LOC" % (
            risk_owner["name"], risk_owner["sole_files"], risk_owner["sole_loc"]))
    print("")
    print("Top knowledge-risk files:")
    for item in data["files"][:5]:
        print("  [%5.1f] %s  (%s, %.0f%%, bf=%d)" % (
            item["risk"], item["path"], item["top_author"],
            item["top_share"], item["bus_factor"]))
    print("")
    print("Report written to %s" % out_html)


def main(argv):
    scope = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else ""
    root = repo_root(scope or ".")
    if root is None:
        sys.stderr.write("not a git repository\n")
        return 2

    rel_scope = ""
    if scope:
        absolute = os.path.abspath(scope)
        rel_scope = os.path.relpath(absolute, root)
        if rel_scope == ".":
            rel_scope = ""

    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    data = analyze(root, rel_scope, now_ts)
    data["repo"] = os.path.basename(root)
    data["scope"] = rel_scope or "(whole repo)"
    data["generated"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out_dir = os.path.join(os.getcwd(), "bus-factor-report")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "data.json")
    out_html = os.path.join(out_dir, "index.html")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    here = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(here, "..", "assets", "template.html")
    template_path = os.path.normpath(template_path)
    if os.path.exists(template_path):
        render(template_path, data, out_html)
    else:
        sys.stderr.write("template not found at %s\n" % template_path)

    print_summary(data, out_html)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
