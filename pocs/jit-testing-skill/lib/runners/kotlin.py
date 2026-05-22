from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile
from lib.runners.base import BaseRunner

KOTLIN_MATRIX = {
    "Int": [0, 1, -1, 100, 499, 500, 4999, 5000],
    "Long": [0, 1, -1, 100, 499, 500, 4999, 5000],
    "Double": [0.0, 1.0, -1.0, 49.99, 50.0],
    "Float": [0.0, 1.0, -1.0, 49.99, 50.0],
    "Boolean": ["true", "false"],
    "String": ['""', '"x"', '"hello"'],
}


class KotlinRunner(BaseRunner):
    target = "kotlin"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=None, max_tests=max_tests)

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=intent, max_tests=max_tests)

    def _behavior_diff(self, repo, diff, rdir, intent, max_tests):
        files = self._diff_files(repo, diff, [".kt"])
        files = [f for f in files if not _is_test_file(f)]
        if not files:
            return []
        if not shutil.which("kotlinc") or not shutil.which("java"):
            return [{"name": "kotlin", "sense_check": "kotlinc or java not on PATH", "kind": "skip"}]
        parent_rev, head_rev = self._parent_head(diff)
        catches = []
        for path in files:
            parent_src = self._read_at_rev(repo, parent_rev, path)
            head_src = self._read_at_rev(repo, head_rev, path)
            if head_src is None or parent_src is None:
                continue
            pkg = _extract_package(head_src)
            funcs = _extract_functions(head_src)
            if not funcs:
                continue
            for fn in funcs[:max_tests]:
                rows = _generate_inputs(fn, max_tests)
                if not rows:
                    continue
                parent_outs = _probe(parent_src, pkg, fn, rows)
                head_outs = _probe(head_src, pkg, fn, rows)
                if parent_outs is None or head_outs is None:
                    continue
                for row, po, ho in zip(rows, parent_outs, head_outs):
                    if po is None or ho is None:
                        continue
                    if po == ho:
                        continue
                    qualified = f"{pkg}.{fn['name']}" if pkg else fn["name"]
                    sense = (
                        f"On the parent, `{qualified}({row['display']})` "
                        f"returned `{po}`. On your change, it returns `{ho}`. Is this expected?"
                    )
                    test_code = _render_test(pkg, fn, row, po)
                    tname = f"test_{fn['name']}_{abs(hash(row['display'])) % 10000}.kt"
                    (rdir / "tests" / tname).write_text(test_code)
                    catches.append({
                        "name": qualified,
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
    return "/test/" in path or path.endswith("Test.kt") or path.endswith("Spec.kt")


def _extract_package(src):
    m = re.search(r"^\s*package\s+([\w.]+)", src, re.MULTILINE)
    return m.group(1) if m else None


def _extract_functions(src):
    out = []
    pat = re.compile(
        r"\bfun\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([\w<>?., ]+))?",
        re.MULTILINE,
    )
    for m in pat.finditer(src):
        name = m.group(1)
        if name.startswith("_") or name == "main":
            continue
        params = []
        for part in m.group(2).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                pname, ptype = part.split(":", 1)
            else:
                pname, ptype = part, "Int"
            params.append({"name": pname.strip(), "type": ptype.strip()})
        out.append({"name": name, "params": params, "return": (m.group(3) or "Any").strip()})
    return out


def _generate_inputs(fn, max_rows):
    if not fn["params"]:
        return [{"display": "", "args": []}]
    matrices = []
    for p in fn["params"]:
        vals = KOTLIN_MATRIX.get(p["type"], KOTLIN_MATRIX["Int"])
        matrices.append(vals)
    rows = [[]]
    for col in matrices:
        rows = [r + [v] for r in rows for v in col]
        if len(rows) > max_rows * 5:
            rows = rows[: max_rows * 5]
    out = []
    for r in rows[:max_rows]:
        out.append({"display": ", ".join(str(v) for v in r), "args": r})
    return out


def _probe(src, pkg, fn, rows):
    qualified = f"{pkg}.{fn['name']}" if pkg else fn["name"]
    parts = []
    for i, row in enumerate(rows):
        args = ", ".join(str(v) for v in row["args"])
        parts.append(
            "  try {\n"
            f"    val v = {qualified}({args})\n"
            f'    println("__JIT__{{\\"i\\":{i},\\"ok\\":true,\\"value\\":\\"${{v}}\\"}}")\n'
            "  } catch (e: Throwable) {\n"
            f'    println("__JIT__{{\\"i\\":{i},\\"ok\\":false,\\"err\\":\\"${{e.javaClass.simpleName}}\\"}}")\n'
            "  }\n"
        )
    probe_body = "fun main() {\n" + "".join(parts) + "}\n"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "Target.kt").write_text(src)
        (d / "Probe.kt").write_text(probe_body)
        jar = d / "probe.jar"
        try:
            rc = subprocess.run(
                ["kotlinc", str(d / "Target.kt"), str(d / "Probe.kt"),
                 "-include-runtime", "-d", str(jar), "-nowarn"],
                capture_output=True, text=True, timeout=240,
            )
        except Exception:
            return None
        if rc.returncode != 0:
            return None
        try:
            rj = subprocess.run(
                ["java", "-jar", str(jar)],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            return None
    out_by_i = {}
    for line in (rj.stdout or "").splitlines():
        marker = "__JIT__"
        idx = line.find(marker)
        if idx < 0:
            continue
        payload = line[idx + len(marker):]
        try:
            data = json.loads(payload)
        except Exception:
            continue
        out_by_i[data["i"]] = data.get("value") if data.get("ok") else data.get("err")
    if not out_by_i:
        return None
    return [out_by_i.get(i) for i in range(len(rows))]


def _render_test(pkg, fn, row, expected):
    args = ", ".join(str(v) for v in row["args"])
    qualified = f"{pkg}.{fn['name']}" if pkg else fn["name"]
    pkg_line = f"package {pkg}\n\n" if pkg else ""
    return (
        f"{pkg_line}"
        "import org.junit.jupiter.api.Test\n"
        "import org.junit.jupiter.api.Assertions.assertEquals\n\n"
        f"class {fn['name'].capitalize()}CatchTest {{\n"
        "    @Test\n"
        f"    fun parentBehaviorPreservedFor_{fn['name']}() {{\n"
        f'        assertEquals("{expected}", {qualified}({args}).toString())\n'
        "    }\n"
        "}\n"
    )
