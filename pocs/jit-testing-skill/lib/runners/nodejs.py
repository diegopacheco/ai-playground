from pathlib import Path
import subprocess
import tempfile
import re
import json
import shutil
from lib.runners.base import BaseRunner

NODE_INPUT_MATRIX = {
    "number": [0, 1, -1, 100, 499, 500, 4999, 5000, 100000],
    "boolean": [True, False],
    "string": ["", "x", "hello"],
}

class NodejsRunner(BaseRunner):
    target = "nodejs"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=None, max_tests=max_tests)

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=intent, max_tests=max_tests)

    def _behavior_diff(self, repo, diff, rdir, intent, max_tests):
        js_files = self._diff_files(repo, diff, [".js", ".mjs", ".cjs"])
        js_files = [f for f in js_files if not _is_test_file(f)]
        if not js_files:
            return []
        if not shutil.which("node"):
            return [{"name": "nodejs", "sense_check": "node binary not found", "kind": "skip"}]
        parent_rev, head_rev = self._parent_head(diff)
        catches = []
        for path in js_files:
            parent_src = self._read_at_rev(repo, parent_rev, path)
            head_src = self._read_at_rev(repo, head_rev, path)
            if head_src is None:
                head_src = (repo / path).read_text()
            if parent_src is None:
                continue
            funcs = _extract_functions(head_src)
            if not funcs:
                continue
            with tempfile.TemporaryDirectory() as pd, tempfile.TemporaryDirectory() as hd:
                pdir = Path(pd)
                hdir = Path(hd)
                pf = pdir / Path(path).name
                hf = hdir / Path(path).name
                pf.write_text(parent_src)
                hf.write_text(head_src)
                for fn in funcs[:max_tests]:
                    rows = _generate_inputs(fn, max_tests)
                    for row in rows:
                        po = _invoke_node(pf, fn["name"], row)
                        ho = _invoke_node(hf, fn["name"], row)
                        if po is None or ho is None:
                            continue
                        if po == ho:
                            continue
                        sense = (
                            f"On the parent, `{fn['name']}({row['display']})` "
                            f"returned `{po}`. On your change, it returns `{ho}`. Is this expected?"
                        )
                        test_code = _render_node_test(Path(path).name, fn, row, po)
                        tname = f"test_{fn['name']}_{abs(hash(row['display']))%10000}.js"
                        (rdir / "tests" / tname).write_text(test_code)
                        catches.append({
                            "name": fn["name"],
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
    return ".test." in name or name.startswith("test.") or "/test/" in path or "/tests/" in path or "/__tests__/" in path

def _extract_functions(src):
    out = []
    seen = set()
    for m in re.finditer(r"(?:^|\n)\s*(?:module\.exports\.|exports\.)(\w+)\s*=\s*(?:function|\()", src):
        name = m.group(1)
        if name not in seen and not name.startswith("_"):
            out.append({"name": name, "params": _infer_params(src, name)})
            seen.add(name)
    for m in re.finditer(r"function\s+(\w+)\s*\(([^)]*)\)", src):
        name = m.group(1)
        if name in seen or name.startswith("_"):
            continue
        params = _parse_params(m.group(2))
        out.append({"name": name, "params": params})
        seen.add(name)
    return out

def _infer_params(src, name):
    m = re.search(rf"(?:function\s+{re.escape(name)}|{re.escape(name)}\s*=\s*(?:function)?\s*)\(([^)]*)\)", src)
    if not m:
        return []
    return _parse_params(m.group(1))

def _parse_params(s):
    params = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        pname = part.split("=")[0].strip()
        params.append({"name": pname, "type": "number"})
    return params

def _generate_inputs(fn, max_rows):
    if not fn["params"]:
        return [{"display": "", "args": []}]
    matrices = []
    for p in fn["params"]:
        if p["name"].lower() in ("pickup", "flag", "enabled", "is", "isactive"):
            matrices.append(NODE_INPUT_MATRIX["boolean"])
        elif p["name"].lower() in ("name", "id", "key"):
            matrices.append(NODE_INPUT_MATRIX["string"])
        else:
            matrices.append(NODE_INPUT_MATRIX["number"])
    rows = [[]]
    for col in matrices:
        rows = [r + [v] for r in rows for v in col]
        if len(rows) > max_rows * 5:
            rows = rows[: max_rows * 5]
    out = []
    for r in rows[:max_rows]:
        out.append({"display": ", ".join(json.dumps(v) for v in r), "args": r})
    return out

def _invoke_node(file: Path, fn: str, row):
    args_json = json.dumps(row["args"])
    probe = f"""
const path = {str(file)!r};
const mod = require(path);
const args = {args_json};
try {{
    const f = (mod && mod.{fn}) ? mod.{fn} : (typeof {fn} !== 'undefined' ? {fn} : null);
    if (!f) {{ process.stdout.write(JSON.stringify({{ok:false,err:'no_export'}})); process.exit(0); }}
    const r = f.apply(null, args);
    process.stdout.write(JSON.stringify({{ok:true, value: JSON.stringify(r)}}));
}} catch (e) {{
    process.stdout.write(JSON.stringify({{ok:false,err:String(e)}}));
}}
"""
    rp = subprocess.run(
        ["node", "-e", probe],
        capture_output=True, text=True, timeout=10,
    )
    if rp.returncode != 0:
        return None
    try:
        data = json.loads(rp.stdout.strip())
    except Exception:
        return None
    if not data.get("ok"):
        return data.get("err")
    return data.get("value")

def _render_node_test(filename, fn, row, expected):
    args = ", ".join(json.dumps(v) for v in row["args"])
    return f"""const {{ {fn['name']} }} = require('./{filename}');
const assert = require('assert');

assert.strictEqual(JSON.stringify({fn['name']}({args})), {json.dumps(expected)});
"""
