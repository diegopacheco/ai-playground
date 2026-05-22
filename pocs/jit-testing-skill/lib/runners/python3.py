from pathlib import Path
import subprocess
import tempfile
import re
import json
import sys
from lib.runners.base import BaseRunner

PY_INPUT_MATRIX = {
    "int": [0, 1, -1, 100, 499, 500, 4999, 5000, 100000],
    "float": [0.0, 1.0, -1.0, 49.99, 50.0],
    "bool": [True, False],
    "str": ["", "x", "hello"],
}

class Python3Runner(BaseRunner):
    target = "python3"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=None, max_tests=max_tests)

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=intent, max_tests=max_tests)

    def _behavior_diff(self, repo, diff, rdir, intent, max_tests):
        py_files = self._diff_files(repo, diff, [".py"])
        py_files = [f for f in py_files if not _is_test_file(f)]
        if not py_files:
            return []
        parent_rev, head_rev = self._parent_head(diff)
        catches = []
        for path in py_files:
            parent_src = self._read_at_rev(repo, parent_rev, path)
            head_src = self._read_at_rev(repo, head_rev, path)
            if head_src is None:
                head_src = (repo / path).read_text()
            if parent_src is None:
                continue
            functions = _extract_functions(head_src)
            if not functions:
                continue
            with tempfile.TemporaryDirectory() as pd, tempfile.TemporaryDirectory() as hd:
                parent_root = Path(pd)
                head_root = Path(hd)
                parent_target = parent_root / Path(path).name
                head_target = head_root / Path(path).name
                parent_target.write_text(parent_src)
                head_target.write_text(head_src)
                module_name = Path(path).stem
                for fn in functions[:max_tests]:
                    rows = _generate_inputs(fn, max_rows=max_tests)
                    for row in rows:
                        po = _invoke_py(parent_root, module_name, fn["name"], row)
                        ho = _invoke_py(head_root, module_name, fn["name"], row)
                        if po is None or ho is None:
                            continue
                        if po == ho:
                            continue
                        sense = (
                            f"On the parent, `{module_name}.{fn['name']}({row['display']})` "
                            f"returned `{po}`. On your change, it returns `{ho}`. Is this expected?"
                        )
                        test_code = _render_pytest(module_name, fn, row, po)
                        tname = f"test_{module_name}_{fn['name']}_{abs(hash(row['display']))%10000}.py"
                        (rdir / "tests" / tname).write_text(test_code)
                        catches.append({
                            "name": f"{module_name}.{fn['name']}",
                            "sense_check": sense,
                            "behavior_input": row["display"],
                            "parent_output": po,
                            "diff_output": ho,
                            "test_code": test_code,
                            "trace": f"expected <{po}> but was <{ho}>",
                            "kind": "behavior_diff",
                        })
        return catches

def _is_test_file(path):
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in path

def _extract_functions(src):
    out = []
    pat = re.compile(r"^def\s+(\w+)\s*\(([^)]*)\)\s*(->\s*[\w\[\], ]+)?\s*:", re.MULTILINE)
    for m in pat.finditer(src):
        name = m.group(1)
        if name.startswith("_"):
            continue
        params = []
        for part in m.group(2).split(","):
            part = part.strip()
            if not part or part == "self":
                continue
            pname = part.split(":")[0].split("=")[0].strip()
            ptype = "int"
            if ":" in part:
                ptype = part.split(":", 1)[1].split("=")[0].strip()
            params.append({"name": pname, "type": ptype})
        out.append({"name": name, "params": params})
    return out

def _generate_inputs(fn, max_rows):
    if not fn["params"]:
        return [{"display": "", "args": []}]
    matrices = []
    for p in fn["params"]:
        t = p["type"]
        vals = PY_INPUT_MATRIX.get(t, PY_INPUT_MATRIX["int"])
        matrices.append(vals)
    rows = [[]]
    for col in matrices:
        rows = [r + [v] for r in rows for v in col]
        if len(rows) > max_rows * 5:
            rows = rows[: max_rows * 5]
    out = []
    for r in rows[:max_rows]:
        out.append({"display": ", ".join(repr(v) for v in r), "args": r})
    return out

def _invoke_py(root: Path, module: str, fn: str, row):
    args_repr = ", ".join(repr(v) for v in row["args"])
    probe = f"""
import json, sys, traceback
sys.path.insert(0, {str(root)!r})
try:
    import {module} as m
    result = m.{fn}({args_repr})
    print(json.dumps({{"ok": True, "value": repr(result)}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}))
"""
    rp = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=10,
    )
    if rp.returncode != 0:
        return None
    try:
        data = json.loads(rp.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    if not data.get("ok"):
        return data.get("error")
    return data.get("value")

def _render_pytest(module, fn, row, expected):
    args = ", ".join(repr(v) for v in row["args"])
    return f"""import {module}

def test_parent_behavior_preserved_for_{fn['name']}():
    assert repr({module}.{fn['name']}({args})) == {expected!r}
"""
