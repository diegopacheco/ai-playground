from pathlib import Path
import subprocess
import tempfile
import shutil
import re
import json
from lib.runners.base import BaseRunner

INPUT_MATRIX = {
    "int": ["0", "1", "-1", "100", "499", "500", "4999", "5000", "100000"],
    "long": ["0L", "1L", "-1L", "5000L"],
    "boolean": ["true", "false"],
    "String": ["\"\"", "\"x\"", "\"hello\""],
    "double": ["0.0", "1.0", "-1.0", "49.99", "50.0"],
}

class Java8Runner(BaseRunner):
    target = "java8"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=None, max_tests=max_tests)

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=intent, max_tests=max_tests)

    def _collect_companions(self, repo, head_rev, target_path, parent_rev):
        if self.mode == "snapshot":
            return _collect_companions_snapshot(self, repo, target_path)
        return _collect_companions_git(repo, head_rev, target_path, parent_rev)

    def _behavior_diff(self, repo, diff, rdir, intent, max_tests):
        java_files = self._diff_files(repo, diff, [".java"])
        if not java_files:
            return []
        parent_rev, head_rev = self._parent_head(diff)
        catches = []
        for path in java_files:
            parent_src = self._read_at_rev(repo, parent_rev, path)
            head_src = self._read_at_rev(repo, head_rev, path) or (repo / path).read_text()
            if not parent_src or not head_src:
                continue
            cls = _extract_class(head_src)
            if not cls:
                continue
            methods = _extract_methods(head_src)
            if not methods:
                continue
            companions = self._collect_companions(repo, head_rev, path, parent_rev)
            with tempfile.TemporaryDirectory() as parent_dir, tempfile.TemporaryDirectory() as head_dir:
                parent_root = Path(parent_dir)
                head_root = Path(head_dir)
                _write_tree(parent_root, path, parent_src, companions, use_parent=True)
                _write_tree(head_root, path, head_src, companions, use_parent=False)
                if not _javac(parent_root) or not _javac(head_root):
                    continue
                for method in methods[:max_tests]:
                    rows = _generate_inputs(method, head_src, max_rows=max_tests)
                    for row in rows:
                        parent_out = _invoke(parent_root, cls, method["name"], row, method)
                        head_out = _invoke(head_root, cls, method["name"], row, method)
                        if parent_out is None or head_out is None:
                            continue
                        if parent_out == head_out:
                            continue
                        sense = (
                            f"On the parent, `{cls}.{method['name']}({row['display']})` "
                            f"returned `{parent_out}`. On your change, it returns `{head_out}`. Is this expected?"
                        )
                        test_code = _render_junit(cls, method, row, parent_out)
                        tests_dir = rdir / "tests"
                        tname = f"{cls}_{method['name']}_{abs(hash(row['display']))%10000}.java"
                        (tests_dir / tname).write_text(test_code)
                        catches.append({
                            "name": f"{cls}.{method['name']}",
                            "sense_check": sense,
                            "behavior_input": row["display"],
                            "parent_output": parent_out,
                            "diff_output": head_out,
                            "test_code": test_code,
                            "trace": f"expected <{parent_out}> but was <{head_out}>",
                            "kind": "behavior_diff",
                        })
        return catches

def _extract_class(src):
    m = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", src)
    if m:
        return m.group(1)
    m = re.search(r"public\s+(?:final\s+)?record\s+(\w+)", src)
    return m.group(1) if m else None

def _extract_methods(src):
    pat = re.compile(
        r"public\s+(?!class\b|record\b|interface\b)([\w<>\[\],\s\?]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{",
        re.MULTILINE,
    )
    methods = []
    for m in pat.finditer(src):
        ret = m.group(1).strip()
        name = m.group(2).strip()
        params_raw = m.group(3).strip()
        if name in ("if", "for", "while", "switch"):
            continue
        params = []
        if params_raw:
            for part in _split_params(params_raw):
                tokens = part.strip().split()
                if len(tokens) < 2:
                    continue
                ptype = " ".join(tokens[:-1])
                pname = tokens[-1]
                params.append({"type": ptype, "name": pname})
        if ret == "void":
            continue
        methods.append({"name": name, "params": params, "ret": ret})
    return methods

def _split_params(s):
    out, depth, buf = [], 0, []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out

def _collect_companions_snapshot(runner, repo, target_path):
    tdir = str(Path(target_path).parent)
    out = []
    pdir = Path(repo) / runner.parent_dir
    if not pdir.exists():
        return out
    for f in pdir.rglob("*.java"):
        rel = str(f.relative_to(pdir))
        if rel == target_path:
            continue
        if str(Path(rel).parent) != tdir:
            continue
        parent_src = f.read_text()
        head_path = Path(repo) / rel
        head_src = head_path.read_text() if head_path.exists() else parent_src
        out.append({"path": rel, "head": head_src, "parent": parent_src})
    head_root = Path(repo)
    for f in head_root.rglob("*.java"):
        try:
            rel = str(f.relative_to(head_root))
        except ValueError:
            continue
        if rel.startswith(runner.parent_dir + "/") or rel.startswith(".jit-testing/"):
            continue
        if rel == target_path:
            continue
        if str(Path(rel).parent) != tdir:
            continue
        if any(c["path"] == rel for c in out):
            continue
        head_src = f.read_text()
        parent_path = pdir / rel
        parent_src = parent_path.read_text() if parent_path.exists() else head_src
        out.append({"path": rel, "head": head_src, "parent": parent_src})
    return out

def _collect_companions_git(repo, head_rev, target_path, parent_rev):
    tdir = str(Path(target_path).parent)
    out = []
    try:
        listing = subprocess.check_output(
            ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", head_rev, "--"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return out
    for line in listing.splitlines():
        if not line.endswith(".java"):
            continue
        if line == target_path:
            continue
        if str(Path(line).parent) != tdir:
            continue
        try:
            head_src = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{head_rev}:{line}"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except Exception:
            head_src = None
        try:
            parent_src = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{parent_rev}:{line}"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except Exception:
            parent_src = head_src
        out.append({"path": line, "head": head_src, "parent": parent_src})
    return out

def _write_tree(root: Path, target_path: str, target_src: str, companions, use_parent: bool):
    p = root / target_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(target_src)
    for c in companions:
        cp = root / c["path"]
        cp.parent.mkdir(parents=True, exist_ok=True)
        text = c["parent"] if use_parent else c["head"]
        if text:
            cp.write_text(text)

def _javac(root: Path) -> bool:
    files = [str(p) for p in root.rglob("*.java")]
    if not files:
        return False
    proc = subprocess.run(
        ["javac", "-d", str(root / "_out"), "-source", "1.8", "-target", "1.8", *files],
        capture_output=True, text=True,
    )
    return proc.returncode == 0

def _generate_inputs(method, src, max_rows):
    if not method["params"]:
        return [{"display": "()", "args": [], "types": []}]
    matrices = []
    for p in method["params"]:
        t = p["type"]
        if t in INPUT_MATRIX:
            matrices.append([(t, v) for v in INPUT_MATRIX[t]])
        elif _is_record_type(t, src):
            matrices.append(_record_values(t, src))
        else:
            matrices.append([(t, "null")])
    rows = [[]]
    for col in matrices:
        rows = [r + [c] for r in rows for c in col]
        if len(rows) > max_rows * 5:
            rows = rows[: max_rows * 5]
    out = []
    for r in rows[:max_rows]:
        args = [v for _, v in r]
        display = ",".join(args)
        out.append({"display": display, "args": args, "types": [t for t, _ in r]})
    return out

def _is_record_type(t, src):
    return re.search(rf"\brecord\s+{re.escape(t)}\s*\(", src) is not None or re.search(rf"\brecord\s+{re.escape(t)}\s*\(", "") is not None

def _record_values(t, src):
    m = re.search(rf"record\s+{re.escape(t)}\s*\(([^)]*)\)", src)
    if not m:
        return [(t, f"new {t}(0, false)")]
    params = []
    for part in _split_params(m.group(1)):
        toks = part.strip().split()
        if len(toks) >= 2:
            params.append(toks[0])
    if not params:
        return [(t, f"new {t}()")]
    rows = [[]]
    for ptype in params:
        vals = INPUT_MATRIX.get(ptype, ["0"])
        rows = [r + [v] for r in rows for v in vals]
        if len(rows) > 20:
            rows = rows[:20]
    return [(t, f"new {t}({', '.join(r)})") for r in rows]

def _invoke(root: Path, cls, method_name, row, method):
    probe = f"""
public class _Probe {{
    public static void main(String[] args) throws Exception {{
        {cls} __target;
        try {{ __target = new {cls}(); }} catch (Throwable t) {{ __target = null; }}
        Object result;
        if (__target == null) {{
            result = {cls}.{method_name}({", ".join(row["args"])});
        }} else {{
            result = __target.{method_name}({", ".join(row["args"])});
        }}
        System.out.print(String.valueOf(result));
    }}
}}
"""
    probe_path = root / "_Probe.java"
    probe_path.write_text(probe)
    cp = subprocess.run(
        ["javac", "-d", str(root / "_out"), "-source", "1.8", "-target", "1.8",
         "-cp", str(root / "_out"), str(probe_path)],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        return None
    rp = subprocess.run(
        ["java", "-cp", str(root / "_out"), "_Probe"],
        capture_output=True, text=True, timeout=10,
    )
    if rp.returncode != 0:
        return None
    return rp.stdout.strip()

def _render_junit(cls, method, row, expected):
    args = ", ".join(row["args"])
    return f"""import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class {cls}CatchTest {{
    @Test
    void parent_behavior_preserved_for_{method['name']}() {{
        {cls} __t = new {cls}();
        assertEquals("{expected}", String.valueOf(__t.{method['name']}({args})));
    }}
}}
"""
