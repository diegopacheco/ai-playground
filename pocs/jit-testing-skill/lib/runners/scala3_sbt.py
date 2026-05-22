from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile
from lib.runners.base import BaseRunner

SCALA_MATRIX = {
    "Int": [0, 1, -1, 100, 499, 500, 4999, 5000],
    "Long": [0, 1, -1, 100, 499, 500, 4999, 5000],
    "Double": [0.0, 1.0, -1.0, 49.99, 50.0],
    "Float": [0.0, 1.0, -1.0, 49.99, 50.0],
    "Boolean": ["true", "false"],
    "String": ['""', '"x"', '"hello"'],
}


class Scala3SbtRunner(BaseRunner):
    target = "scala3-sbt"
    scala_version = "3"

    def dodgy_diff(self, repo, diff, rdir, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=None, max_tests=max_tests)

    def intent_aware(self, repo, diff, rdir, intent, max_tests):
        return self._behavior_diff(repo, diff, rdir, intent=intent, max_tests=max_tests)

    def _behavior_diff(self, repo, diff, rdir, intent, max_tests):
        files = self._diff_files(repo, diff, [".scala"])
        files = [f for f in files if not _is_test_file(f)]
        if not files:
            return []
        if not shutil.which("scala-cli"):
            return [{"name": "scala3-sbt", "sense_check": "scala-cli not on PATH", "kind": "skip"}]
        parent_rev, head_rev = self._parent_head(diff)
        catches = []
        for path in files:
            parent_src = self._read_at_rev(repo, parent_rev, path)
            head_src = self._read_at_rev(repo, head_rev, path)
            if head_src is None or parent_src is None:
                continue
            pkg = _extract_package(head_src)
            obj = _extract_object(head_src)
            funcs = _extract_functions(head_src)
            if not funcs or not obj:
                continue
            for fn in funcs[:max_tests]:
                rows = _generate_inputs(fn, max_tests)
                if not rows:
                    continue
                parent_outs = _probe(parent_src, pkg, obj, fn, rows, self.scala_version)
                head_outs = _probe(head_src, pkg, obj, fn, rows, self.scala_version)
                if parent_outs is None or head_outs is None:
                    continue
                for row, po, ho in zip(rows, parent_outs, head_outs):
                    if po is None or ho is None:
                        continue
                    if po == ho:
                        continue
                    sense = (
                        f"On the parent, `{obj}.{fn['name']}({row['display']})` "
                        f"returned `{po}`. On your change, it returns `{ho}`. Is this expected?"
                    )
                    test_code = _render_test(pkg, obj, fn, row, po, self.scala_version)
                    tname = f"test_{obj}_{fn['name']}_{abs(hash(row['display'])) % 10000}.scala"
                    (rdir / "tests" / tname).write_text(test_code)
                    catches.append({
                        "name": f"{obj}.{fn['name']}",
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
    return "/test/" in path or path.endswith("Test.scala") or path.endswith("Spec.scala")


def _extract_package(src):
    m = re.search(r"^\s*package\s+([\w.]+)", src, re.MULTILINE)
    return m.group(1) if m else None


def _extract_object(src):
    m = re.search(r"\bobject\s+(\w+)", src)
    return m.group(1) if m else None


def _extract_functions(src):
    out = []
    pat = re.compile(r"\bdef\s+(\w+)\s*\(([^)]*)\)\s*:\s*([\w\[\], ]+?)\s*=", re.MULTILINE)
    for m in pat.finditer(src):
        name = m.group(1)
        if name.startswith("_"):
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
        out.append({"name": name, "params": params, "return": m.group(3).strip()})
    return out


def _generate_inputs(fn, max_rows):
    if not fn["params"]:
        return [{"display": "", "args": []}]
    matrices = []
    for p in fn["params"]:
        vals = SCALA_MATRIX.get(p["type"], SCALA_MATRIX["Int"])
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


def _probe(src, pkg, obj, fn, rows, scala_version):
    qualified = f"{pkg}.{obj}" if pkg else obj
    parts = []
    for i, row in enumerate(rows):
        args = ", ".join(str(v) for v in row["args"])
        parts.append(
            "  try {\n"
            f"    val v = {qualified}.{fn['name']}({args})\n"
            f'    println(s"""__JIT__{{"i":{i},"ok":true,"value":"${{v}}"}}""")\n'
            "  } catch { case e: Throwable =>\n"
            f'    println(s"""__JIT__{{"i":{i},"ok":false,"err":"${{e.getClass.getSimpleName}}"}}""")\n'
            "  }\n"
        )
    if scala_version.startswith("3"):
        directive = f"//> using scala {scala_version}\n"
        probe_body = "@main def jitProbe(): Unit =\n" + "".join(parts)
    else:
        directive = f"//> using scala {scala_version}\n"
        probe_body = (
            "object JitProbe {\n"
            "  def main(args: Array[String]): Unit = {\n"
            + "".join("  " + line + "\n" for line in "".join(parts).splitlines())
            + "  }\n}\n"
        )
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "Target.scala").write_text(src)
        (d / "Probe.scala").write_text(directive + probe_body)
        try:
            r = subprocess.run(
                ["scala-cli", "run", str(d), "-q"],
                capture_output=True, text=True, timeout=180,
            )
        except Exception:
            return None
    out_by_i = {}
    for line in (r.stdout or "").splitlines():
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


def _render_test(pkg, obj, fn, row, expected, scala_version):
    args = ", ".join(str(v) for v in row["args"])
    qualified = f"{pkg}.{obj}" if pkg else obj
    return (
        f"//> using scala {scala_version}\n"
        f'//> using test.dep org.scalameta::munit::1.0.0\n\n'
        f"class {obj}CatchSuite extends munit.FunSuite {{\n"
        f'  test("parent behavior preserved for {fn["name"]}") {{\n'
        f'    assertEquals({qualified}.{fn["name"]}({args}).toString, "{expected}")\n'
        f"  }}\n"
        f"}}\n"
    )
