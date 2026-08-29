#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

SKIP_DIRS = {
    ".git", "node_modules", "target", "build", "dist", "out", "vendor",
    ".venv", "venv", "__pycache__", ".idea", ".vscode", ".gradle", ".next",
    "coverage", ".pytest_cache", ".mypy_cache", "obj", ".terraform",
    "site-packages", ".cache", ".tox", "Pods", "DerivedData", ".svelte-kit",
    "playwright-report", "test-results", ".turbo", ".parcel-cache", ".angular",
    "storybook-static", "htmlcov", ".nyc_output", "__snapshots__", ".dart_tool",
}
SKIP_FILE_SUFFIX = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".gz",
    ".jar", ".war", ".class", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".lock", ".min.js",
    ".map", ".pyc", ".wasm",
)
LANGS = {
    ".java": "Java", ".kt": "Kotlin", ".scala": "Scala", ".py": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
    ".php": "PHP", ".cs": "C#", ".c": "C", ".h": "C", ".cpp": "C++",
    ".hpp": "C++", ".swift": "Swift", ".sh": "Shell", ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".vue": "Vue",
    ".ex": "Elixir", ".clj": "Clojure", ".lua": "Lua",
}
CODE_EXT = set(LANGS)
BUILD_FILES = {
    "pom.xml": "maven", "build.gradle": "gradle", "build.gradle.kts": "gradle",
    "package.json": "npm", "go.mod": "go", "Cargo.toml": "cargo",
    "pyproject.toml": "python", "setup.py": "python", "Gemfile": "ruby",
    "composer.json": "php", "build.sbt": "sbt", "mix.exs": "elixir",
    "CMakeLists.txt": "cmake", "requirements.txt": "python",
}
MAX_BYTES = 900_000
BIG_FILE_LINES = 400
LONG_FUNC_LINES = 60


def run(cmd, cwd):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return ""


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(SKIP_FILE_SUFFIX):
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.islink(full) or os.path.getsize(full) > MAX_BYTES:
                    continue
            except OSError:
                continue
            yield full, os.path.relpath(full, root).replace(os.sep, "/")


def github_url(root):
    remote = run(["git", "remote", "get-url", "origin"], root).strip()
    if not remote:
        return None, None
    url = remote
    if url.startswith("git@"):
        url = "https://" + url[4:].replace(":", "/", 1)
    if url.endswith(".git"):
        url = url[:-4]
    if "github.com" not in url:
        return None, remote
    return url, remote


TEST_RULES = [
    ("e2e", [r"(^|/)(e2e|cypress|playwright|selenium)(/|$)", r"\.e2e\.", r"_e2e_", r"spec\.cy\."]),
    ("chaos", [r"chaos", r"toxiproxy", r"litmus", r"gremlin"]),
    ("stress", [r"(^|/)(k6|gatling|jmeter|locust)(/|$)", r"stress", r"load[-_]?test", r"benchmark_load"]),
    ("benchmark", [r"bench", r"jmh"]),
    ("property", [r"hypothesis", r"quickcheck", r"jqwik", r"fast-check", r"proptest", r"scalacheck"]),
    ("css", [r"visual[-_]?regression", r"percy", r"storyshot", r"snapshot.*\.css", r"chromatic"]),
    ("database", [r"repositor(y|ies)test", r"dao.*test", r"testcontainer", r"flyway.*test", r"\bdbtest"]),
    ("contract", [r"pact", r"contract[-_]?test", r"wiremock", r"spring-cloud-contract"]),
    ("smoke", [r"smoke"]),
    ("integration", [r"(^|/)it(/|$)", r"integration", r"\bIT\.java$", r"_it\.", r"\.it\."]),
    ("unit", [r"(^|/)test(s)?(/|$)", r"__tests__", r"\btest_", r"_test\.", r"\.test\.", r"\.spec\.", r"Test\.(java|kt|scala|cs)$", r"Spec\.(java|kt|scala)$"]),
]
CASE_PATTERNS = [
    r"@Test\b", r"@ParameterizedTest\b", r"@RepeatedTest\b", r"#\[test\]",
    r"\bfunc\s+Test[A-Z]", r"\bdef\s+test_", r"\bit\s*\(", r"\btest\s*\(",
    r"#\[[\w:]*test\]",
    r"\bit\.each\b", r"\bscenario\s*\(", r"\bshould\s*\(", r"\bQuickCheck\b",
    r"\bt\.Run\s*\(",
]
LOG_PATTERNS = {
    "error": [r"\blog(ger)?\.error\s*\(", r"\bconsole\.error\s*\(", r"\blogging\.error\s*\(", r"\blog\.Error", r"\berror!\s*\("],
    "warn": [r"\blog(ger)?\.warn(ing)?\s*\(", r"\bconsole\.warn\s*\(", r"\blogging\.warn", r"\bwarn!\s*\("],
    "info": [r"\blog(ger)?\.info\s*\(", r"\bconsole\.info\s*\(", r"\blogging\.info", r"\binfo!\s*\("],
    "debug": [r"\blog(ger)?\.debug\s*\(", r"\bconsole\.debug\s*\(", r"\blogging\.debug", r"\bdebug!\s*\("],
    "trace": [r"\blog(ger)?\.trace\s*\(", r"\btrace!\s*\("],
    "print": [r"\bconsole\.log\s*\(", r"^\s*print\s*\(", r"\bSystem\.out\.print", r"\bfmt\.Print", r"\bprintln!\s*\("],
}
VALIDATION_PATTERNS = [
    r"@Valid\b", r"@NotNull\b", r"@NotBlank\b", r"@Size\s*\(", r"@Pattern\s*\(",
    r"\bzod\b", r"\bjoi\b", r"\byup\b", r"\bpydantic\b", r"BaseModel",
    r"\bassert\s", r"Objects\.requireNonNull", r"\bvalidate[A-Z_]", r"\bis_valid\b",
    r"raise\s+ValueError", r"throw\s+new\s+IllegalArgumentException",
    r"z\.object\s*\(", r"\.safeParse\s*\(", r"\.parse\s*\(\s*\w+\s*\)",
    r"\bok_or(_else)?\s*\(", r"\bbail!\s*\(", r"\bensure!\s*\(",
    r"\bassert!\s*\(", r"\brequired\s*[:=]\s*true", r"\bcheck[A-Z_]",
    r"\.is_empty\(\)", r"throw\s+new\s+\w*ValidationError",
]
GUARD_PATTERNS = [
    r"\btry\s*[\{:]", r"\bcatch\s*\(", r"\bexcept\b", r"\brescue\b",
    r"\bif\s+err\s*!=\s*nil", r"\.unwrap_or", r"\bResult<", r"\bOption<",
]
EMPTY_CATCH = [r"catch\s*\([^)]*\)\s*\{\s*\}", r"except[^\n:]*:\s*\n\s*pass\b"]
DEBT_MARKERS = r"\b(TODO|FIXME|HACK|XXX|BUG|DEPRECATED|WORKAROUND|TEMP|KLUDGE|REFACTOR)\b"
SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']", "hardcoded credential"),
    (r"AKIA[0-9A-Z]{12,}", "aws access key"),
    (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key"),
]
DASHBOARD_HINTS = ["grafana", "dashboard", "kibana", "datadog"]
ALERT_HINTS = ["alert", "prometheus", "alertmanager", "rules.yml", "rules.yaml", "slo", "pagerduty"]

SQL_CREATE = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?([A-Za-z0-9_.]+)[`\"\]]?\s*\(", re.I)
LIQUIBASE_TABLE = re.compile(r"createTable[^\n]*tableName\s*[:=]\s*[\"']?([A-Za-z0-9_]+)", re.I)
JPA_TABLE = re.compile(r"@Table\s*\(\s*name\s*=\s*\"([^\"]+)\"")
JPA_ENTITY = re.compile(r"@Entity\b")
PRISMA_MODEL = re.compile(r"^\s*model\s+([A-Za-z0-9_]+)\s*\{", re.M)
DJANGO_MODEL = re.compile(r"class\s+([A-Za-z0-9_]+)\s*\(\s*models\.Model")
QUERY_PATTERN = re.compile(r"(SELECT\s+[\s\S]{10,900}?FROM\s+[\s\S]{0,900}?)(?=[\"'`;]|$)", re.I)


FUNC_EXT = {".java", ".kt", ".scala", ".go", ".js", ".jsx", ".ts", ".tsx",
            ".py", ".rs", ".cs", ".rb", ".php", ".swift"}
INDENT_EXT = {".py", ".rb"}
BRACE_FUNC = re.compile(
    r"^[ \t]*(?:@\w+\s*)*(?:public |private |protected |internal |static |final |abstract |async |export |default )*"
    r"(?:function\s+\w+|func\s+\w+|fn\s+\w+|def\s+\w+|[\w<>\[\],. ]+\s+\w+)\s*\([^;]*$", re.M)
INDENT_FUNC = re.compile(r"^([ \t]*)(?:async\s+)?def\s+(\w+)", re.M)


def find_long_functions(text, ext):
    lines = text.splitlines()
    found = []
    if ext in INDENT_EXT:
        for match in INDENT_FUNC.finditer(text):
            start = text[:match.start()].count("\n")
            indent = len(match.group(1).expandtabs(4))
            end = start + 1
            for i in range(start + 1, len(lines)):
                stripped = lines[i].strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if len(lines[i]) - len(lines[i].lstrip().expandtabs(4)) <= indent and \
                        len(lines[i].expandtabs(4)) - len(lines[i].expandtabs(4).lstrip()) <= indent:
                    break
                end = i + 1
            length = end - start
            if length > LONG_FUNC_LINES:
                found.append({"line": start + 1, "lines": length, "name": lines[start].strip()[:120]})
        return found
    for match in BRACE_FUNC.finditer(text):
        start = text[:match.start()].count("\n")
        depth = 0
        opened = False
        end = start
        for i in range(start, min(start + 1200, len(lines))):
            depth += lines[i].count("{") - lines[i].count("}")
            if "{" in lines[i]:
                opened = True
            end = i
            if opened and depth <= 0:
                break
        if not opened:
            continue
        length = end - start + 1
        if length > LONG_FUNC_LINES:
            found.append({"line": start + 1, "lines": length, "name": lines[start].strip()[:120]})
    return found


TEST_EXT = {".java", ".kt", ".scala", ".py", ".js", ".jsx", ".ts", ".tsx",
            ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".ex", ".clj"}


def classify_test(rel, text):
    if os.path.splitext(rel)[1] not in TEST_EXT:
        return None
    low = rel.lower()
    for kind, pats in TEST_RULES:
        for p in pats:
            if re.search(p, low) or re.search(p, rel):
                return kind
    if re.search(r"@Test\b|\bdef test_|\bit\s*\(\s*[\"'`]|\bdescribe\s*\(\s*[\"'`]", text):
        return "unit"
    return None


def count_cases(text):
    return sum(len(re.findall(p, text, re.M)) for p in CASE_PATTERNS)


def detect_modules(root, files):
    modules = {}
    for rel in files:
        base = os.path.basename(rel)
        if base in BUILD_FILES:
            d = os.path.dirname(rel) or "."
            depth = 0 if d == "." else d.count("/") + 1
            if depth > 3:
                continue
            modules.setdefault(d, {"dir": d, "kind": BUILD_FILES[base], "build": rel})
    if len(modules) <= 1:
        for rel in files:
            ext = os.path.splitext(rel)[1]
            if ext not in CODE_EXT:
                continue
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            top = parts[0]
            if top in ("src", "lib", "app", "pkg", "internal", "cmd", "source"):
                top = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
            modules.setdefault(top, {"dir": top, "kind": "directory", "build": None})
    if not modules:
        modules["."] = {"dir": ".", "kind": "directory", "build": None}
    return modules


def module_of(rel, module_dirs):
    best = None
    for d in module_dirs:
        if d == ".":
            continue
        if rel == d or rel.startswith(d + "/"):
            if best is None or len(d) > len(best):
                best = d
    return best or "."


def git_committers(root):
    top = run(["git", "rev-parse", "--show-toplevel"], root).strip()
    prefix = ""
    if top and os.path.abspath(root) != os.path.abspath(top):
        prefix = os.path.relpath(root, top).replace(os.sep, "/") + "/"
    raw = run(["git", "log", "--no-merges", "--pretty=format:%H%x1f%an%x1f%ae%x1f%ad", "--date=short", "--", "."], root)
    if not raw.strip():
        return [], 0
    people = {}
    order = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        sha, name, email, date = parts
        key = email.lower() or name
        if key not in people:
            people[key] = {"name": name, "email": email, "commits": 0,
                           "firstCommit": date, "lastCommit": date,
                           "shas": [], "dirs": Counter(), "subjects": []}
            order.append(key)
        p = people[key]
        p["commits"] += 1
        p["name"] = name
        p["firstCommit"] = min(p["firstCommit"], date)
        p["lastCommit"] = max(p["lastCommit"], date)
        if len(p["shas"]) < 60:
            p["shas"].append(sha)
    total = sum(p["commits"] for p in people.values())
    ranked = sorted(people.values(), key=lambda p: -p["commits"])[:20]
    for p in ranked:
        stat = run(["git", "show", "--pretty=format:%s", "--name-only"] + p["shas"][:40], root)
        for line in stat.splitlines():
            line = line.strip()
            if not line:
                continue
            if "/" in line or "." in line:
                if prefix and not line.startswith(prefix):
                    continue
                scoped = line[len(prefix):] if prefix else line
                d = os.path.dirname(scoped) or scoped
                if d and not any(s in d.split("/") for s in SKIP_DIRS):
                    p["dirs"][d] += 1
            elif len(p["subjects"]) < 12:
                p["subjects"].append(line)
    out = []
    for p in ranked:
        login = None
        m = re.match(r"^\d+\+?([A-Za-z0-9-]+)@users\.noreply\.github\.com$", p["email"])
        if m:
            login = m.group(1)
        out.append({
            "name": p["name"], "email": p["email"], "login": login,
            "commits": p["commits"], "firstCommit": p["firstCommit"],
            "lastCommit": p["lastCommit"],
            "topDirs": [{"dir": d, "changes": c} for d, c in p["dirs"].most_common(8)],
            "recentSubjects": p["subjects"][:8],
        })
    return out, total


def scan(root):
    root = os.path.abspath(root)
    files = []
    file_index = {}
    for full, rel in walk(root):
        files.append(rel)
        file_index[rel] = full

    lang_counter = Counter()
    lang_lines = Counter()
    modules = detect_modules(root, files)
    module_dirs = list(modules)
    for m in modules.values():
        m.update({"files": 0, "lines": 0, "langs": Counter(), "topFiles": [],
                  "tests": 0, "logs": 0, "debtMarks": 0})

    tests = defaultdict(lambda: {"files": 0, "cases": 0, "samples": []})
    logs = Counter()
    log_files = Counter()
    validation_hits = Counter()
    guard_hits = 0
    empty_catches = []
    debt_marks = []
    big_files = []
    long_funcs = []
    secrets = []
    dashboards = []
    alerts = []
    tables = {}
    queries = []
    entities = []
    deps = defaultdict(set)
    configs = []
    externals = Counter()

    for rel in files:
        full = file_index[rel]
        ext = os.path.splitext(rel)[1]
        base = os.path.basename(rel)
        mod = module_of(rel, module_dirs)
        text = read(full)
        if not text:
            continue
        lines = text.count("\n") + 1
        m = modules.get(mod)
        if m is None:
            m = modules.setdefault(mod, {"dir": mod, "kind": "directory", "build": None,
                                         "files": 0, "lines": 0, "langs": Counter(),
                                         "topFiles": [], "tests": 0, "logs": 0, "debtMarks": 0})
        m["files"] += 1
        m["lines"] += lines
        if ext in LANGS:
            lang_counter[LANGS[ext]] += 1
            lang_lines[LANGS[ext]] += lines
            m["langs"][LANGS[ext]] += 1
            m["topFiles"].append((lines, rel))

        if ext in CODE_EXT and lines > BIG_FILE_LINES:
            big_files.append({"path": rel, "lines": lines, "module": mod})

        kind = classify_test(rel, text) if ext in CODE_EXT else None
        if kind:
            cases = count_cases(text)
            t = tests[kind]
            t["files"] += 1
            t["cases"] += cases
            if len(t["samples"]) < 6:
                t["samples"].append({"path": rel, "cases": cases})
            m["tests"] += 1

        for level, pats in LOG_PATTERNS.items():
            hits = sum(len(re.findall(p, text, re.M)) for p in pats)
            if hits:
                logs[level] += hits
                log_files[rel] += hits
                m["logs"] += hits

        v = sum(len(re.findall(p, text)) for p in VALIDATION_PATTERNS)
        if v:
            validation_hits[rel] += v
        guard_hits += sum(len(re.findall(p, text)) for p in GUARD_PATTERNS)
        for p in EMPTY_CATCH:
            for match in re.finditer(p, text):
                empty_catches.append({"path": rel, "line": text[:match.start()].count("\n") + 1})

        for match in re.finditer(DEBT_MARKERS, text):
            line_no = text[:match.start()].count("\n") + 1
            snippet = text.splitlines()[line_no - 1].strip()[:200] if line_no <= lines else ""
            debt_marks.append({"path": rel, "line": line_no, "marker": match.group(1),
                               "snippet": snippet, "module": mod})
            m["debtMarks"] += 1

        for pat, label in SECRET_PATTERNS:
            for match in re.finditer(pat, text):
                secrets.append({"path": rel, "line": text[:match.start()].count("\n") + 1, "kind": label})

        if ext in FUNC_EXT:
            for fn in find_long_functions(text, ext):
                fn.update({"path": rel, "module": mod})
                long_funcs.append(fn)

        low = rel.lower()
        if any(h in low for h in DASHBOARD_HINTS) and ext in (".json", ".yaml", ".yml", ".jsonnet", ".tf"):
            dashboards.append({"path": rel, "type": "grafana" if "grafana" in low else "other"})
        if any(h in low for h in ALERT_HINTS) and ext in (".yaml", ".yml", ".rules", ".tf", ".json"):
            names = re.findall(r"(?:alert|name)\s*:\s*[\"']?([A-Za-z0-9_. -]{3,60})", text)
            alerts.append({"path": rel, "names": names[:20]})

        if base in ("application.yml", "application.yaml", "application.properties",
                    ".env", ".env.example", "config.yml", "config.yaml", "settings.py",
                    "docker-compose.yml", "docker-compose.yaml", "compose.yml"):
            configs.append(rel)
            for host in re.findall(r"(?:https?://|jdbc:|mongodb://|redis://|amqp://|kafka)[A-Za-z0-9_.:/-]{3,60}", text):
                externals[host.split("?")[0][:80]] += 1

        if base == "package.json":
            try:
                pkg = json.loads(text)
                for k in ("dependencies", "devDependencies"):
                    deps[rel].update((pkg.get(k) or {}).keys())
            except Exception:
                pass
        elif base == "pom.xml":
            deps[rel].update(re.findall(r"<artifactId>([^<]+)</artifactId>", text))
        elif base in ("requirements.txt", "Cargo.toml", "go.mod", "build.gradle", "build.gradle.kts"):
            deps[rel].update(re.findall(r"^\s*([A-Za-z0-9_.\-/]{2,60})", text, re.M))

        if ext == ".sql" or "changelog" in low or "migration" in low:
            for match in SQL_CREATE.finditer(text):
                name = match.group(1).split(".")[-1]
                cols = text[match.end():match.end() + 4000].split(";")[0]
                tables.setdefault(name, {"name": name, "source": rel,
                                         "columns": max(1, cols.count(",") + 1) if cols else 0,
                                         "definedBy": "sql"})
        if ext in (".xml", ".yaml", ".yml", ".json") and "changelog" in low:
            for match in LIQUIBASE_TABLE.finditer(text):
                tables.setdefault(match.group(1), {"name": match.group(1), "source": rel,
                                                   "columns": 0, "definedBy": "liquibase"})
        if ext in (".java", ".kt") and JPA_ENTITY.search(text):
            tm = JPA_TABLE.search(text)
            name = tm.group(1) if tm else os.path.splitext(base)[0].lower()
            cols = len(re.findall(r"@Column\b|@Id\b|@JoinColumn\b", text))
            entities.append({"path": rel, "table": name, "columns": cols})
            tables.setdefault(name, {"name": name, "source": rel, "columns": cols, "definedBy": "jpa"})
        if base == "schema.prisma":
            for match in PRISMA_MODEL.finditer(text):
                tables.setdefault(match.group(1), {"name": match.group(1), "source": rel,
                                                   "columns": 0, "definedBy": "prisma"})
        if ext == ".py" and "models" in low:
            for match in DJANGO_MODEL.finditer(text):
                tables.setdefault(match.group(1), {"name": match.group(1), "source": rel,
                                                   "columns": 0, "definedBy": "django"})

        if ext in CODE_EXT or ext == ".sql":
            for match in QUERY_PATTERN.finditer(text):
                q = " ".join(match.group(1).split())
                if len(q) < 24:
                    continue
                flags = []
                if re.search(r"SELECT\s+\*", q, re.I):
                    flags.append("select *")
                if re.search(r"\bJOIN\b", q, re.I) and len(re.findall(r"\bJOIN\b", q, re.I)) >= 3:
                    flags.append("many joins")
                if not re.search(r"\bWHERE\b", q, re.I):
                    flags.append("no where clause")
                has_where = bool(re.search(r"\bWHERE\b", q, re.I))
                has_limit = bool(re.search(r"\bLIMIT\b|\bTOP\b|\bFETCH FIRST\b|\bROWNUM\b", q, re.I))
                if not has_limit and (not has_where or re.search(r"\bORDER\s+BY\b", q, re.I)):
                    flags.append("unbounded")
                if re.search(r"\bLIKE\s+['\"]%", q, re.I):
                    flags.append("leading wildcard")
                if re.search(r"\bOR\b", q, re.I) and re.search(r"\bWHERE\b", q, re.I):
                    flags.append("or in predicate")
                if flags:
                    queries.append({"path": rel, "line": text[:match.start()].count("\n") + 1,
                                    "query": q[:400], "flags": flags})

    for m in modules.values():
        m["topFiles"] = [{"path": p, "lines": n} for n, p in sorted(m["topFiles"], reverse=True)[:12]]
        m["langs"] = dict(m["langs"].most_common(5))

    committers, total_commits = git_committers(root)
    gh, remote = github_url(root)
    dup_deps = Counter()
    for group in deps.values():
        for d in group:
            dup_deps[d] += 1

    data_files = []
    for rel in files:
        if os.path.splitext(rel)[1] in (".db", ".sqlite", ".sqlite3", ".csv"):
            try:
                data_files.append({"path": rel, "bytes": os.path.getsize(file_index[rel])})
            except OSError:
                pass

    total_lines = sum(lang_lines.values())
    log_density = round((sum(logs.values()) / total_lines) * 1000, 2) if total_lines else 0.0

    return {
        "repoRoot": root,
        "repoName": os.path.basename(root),
        "githubUrl": gh,
        "gitRemote": remote,
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root).strip() or None,
        "headCommit": run(["git", "rev-parse", "--short", "HEAD"], root).strip() or None,
        "scannedAt": run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], root).strip(),
        "totals": {
            "files": len(files),
            "codeFiles": sum(lang_counter.values()),
            "lines": total_lines,
            "commits": total_commits,
            "modules": len(modules),
        },
        "languages": [{"name": k, "files": v, "lines": lang_lines[k]} for k, v in lang_counter.most_common()],
        "modules": sorted(modules.values(), key=lambda m: -m["lines"]),
        "committers": committers,
        "tests": {k: v for k, v in sorted(tests.items(), key=lambda kv: -kv[1]["files"])},
        "observability": {
            "logCounts": dict(logs),
            "logDensityPerKLoc": log_density,
            "topLogFiles": [{"path": p, "count": c} for p, c in log_files.most_common(15)],
            "dashboards": dashboards[:30],
            "alerts": alerts[:30],
            "validationHits": [{"path": p, "count": c} for p, c in validation_hits.most_common(20)],
            "validationTotal": sum(validation_hits.values()),
            "guardTotal": guard_hits,
            "emptyCatches": empty_catches[:40],
            "configs": configs[:30],
            "externals": [{"target": k, "count": v} for k, v in externals.most_common(20)],
        },
        "techDebtSignals": {
            "markers": debt_marks[:400],
            "markerCounts": dict(Counter(d["marker"] for d in debt_marks)),
            "bigFiles": sorted(big_files, key=lambda f: -f["lines"])[:40],
            "longFunctions": sorted(long_funcs, key=lambda f: -f["lines"])[:40],
            "secrets": secrets[:40],
            "duplicateDeps": [{"dep": d, "declaredIn": c} for d, c in dup_deps.most_common(20) if c > 1],
        },
        "schema": {
            "tables": sorted(tables.values(), key=lambda t: t["name"]),
            "tableCount": len(tables),
            "entities": entities[:60],
            "suspiciousQueries": sorted(queries, key=lambda q: -len(q["flags"]))[:40],
            "dataFiles": data_files[:20],
        },
        "files": files[:4000],
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "facts.json"
    facts = scan(root)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2)
    t = facts["totals"]
    print(f"scanned {t['files']} files, {t['lines']} lines, {t['modules']} modules, "
          f"{t['commits']} commits -> {out}")


if __name__ == "__main__":
    main()
