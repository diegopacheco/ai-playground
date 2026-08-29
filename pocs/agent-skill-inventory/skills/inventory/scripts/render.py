#!/usr/bin/env python3
import hashlib
import html
import json
import os
import sys
from collections import defaultdict

BOX_H = 56
V_GAP = 124
H_GAP = 46
MARGIN = 46
MAX_PER_ROW = 5
FILL = {
    "module": "#fde8d0", "store": "#d9ecf7", "client": "#e4f5df",
    "queue": "#f6e0ef", "external": "#eee6fb",
}
STROKE = "#6f6a60"
MAX_WHY = 240
MAX_FIX = 330
MAX_DESC_LINES = 5
MIN_DESC_LINES = 3


class Problem(Exception):
    pass


def box_width(label):
    return max(150, min(300, int(len(label) * 9.4) + 36))


def assign_layers(nodes, edges):
    ids = [n["id"] for n in nodes]
    if all(isinstance(n.get("layer"), int) for n in nodes):
        return {n["id"]: n["layer"] for n in nodes}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for e in edges:
        if e["from"] in ids and e["to"] in ids:
            outgoing[e["from"]].append(e["to"])
            incoming[e["to"]].append(e["from"])
    layer = {}
    roots = [i for i in ids if not incoming[i]] or ids[:1]
    frontier = [(r, 0) for r in roots]
    seen = set()
    while frontier:
        node, lv = frontier.pop(0)
        if node in seen and layer.get(node, 0) >= lv:
            continue
        seen.add(node)
        layer[node] = max(layer.get(node, 0), lv)
        if lv < 12:
            for nxt in outgoing[node]:
                frontier.append((nxt, lv + 1))
    for i in ids:
        layer.setdefault(i, 0)
    return layer


def layout(nodes, edges):
    layer = assign_layers(nodes, edges)
    rows = defaultdict(list)
    for n in nodes:
        rows[layer[n["id"]]].append(n)
    ordered = []
    for lv in sorted(rows):
        chunk = rows[lv]
        for i in range(0, len(chunk), MAX_PER_ROW):
            ordered.append(chunk[i:i + MAX_PER_ROW])
    widths = []
    for row in ordered:
        widths.append(sum(box_width(n["label"]) for n in row) + H_GAP * (len(row) - 1))
    canvas_w = max(widths + [520]) + MARGIN * 2
    placed = {}
    y = MARGIN + 26
    for ri, row in enumerate(ordered):
        x = (canvas_w - widths[ri]) / 2
        for n in row:
            w = box_width(n["label"])
            placed[n["id"]] = {"x": x, "y": y, "w": w, "h": BOX_H, "row": ri, "node": n}
            x += w + H_GAP
        y += BOX_H + V_GAP
    canvas_h = y - V_GAP + MARGIN
    return placed, canvas_w, canvas_h, len(ordered)


def anchors(a, b):
    if a["row"] < b["row"]:
        return (a["x"] + a["w"] / 2, a["y"] + a["h"]), (b["x"] + b["w"] / 2, b["y"])
    if a["row"] > b["row"]:
        return (a["x"] + a["w"] / 2, a["y"]), (b["x"] + b["w"] / 2, b["y"] + b["h"])
    if a["x"] < b["x"]:
        return (a["x"] + a["w"], a["y"] + a["h"] / 2), (b["x"], b["y"] + b["h"] / 2)
    return (a["x"], a["y"] + a["h"] / 2), (b["x"] + b["w"], b["y"] + b["h"] / 2)


def diagram(arch):
    nodes = arch.get("nodes") or []
    edges = arch.get("edges") or []
    if not nodes:
        return '<p style="color:#93908a;padding:26px;text-align:center">No architecture nodes were identified.</p>'
    placed, w, h, rows = layout(nodes, edges)
    spans = [abs(placed[e["from"]]["row"] - placed[e["to"]]["row"])
             for e in edges if e["from"] in placed and e["to"] in placed]
    long_spans = [s for s in spans if s > 1]
    if long_spans:
        pad = 260 + 60 * max(long_spans) + 56 * max(0, len(long_spans) - 1)
        for p in placed.values():
            p["x"] += pad
        w += pad * 2
    parts = [
        '__SVG_OPEN__',
        '<defs>',
        '<filter id="wob" x="-12%" y="-12%" width="124%" height="124%">',
        '<feTurbulence type="fractalNoise" baseFrequency="0.022" numOctaves="3" seed="7" result="n"/>',
        '<feDisplacementMap in="SourceGraphic" in2="n" scale="2.4" '
        'xChannelSelector="R" yChannelSelector="G"/></filter>',
        '<marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{STROKE}"/></marker>',
        '</defs>',
        '__SVG_BG__',
        '<g filter="url(#wob)">',
    ]
    same_row_seen = defaultdict(int)
    xs = [p["x"] for p in placed.values()] + [p["x"] + p["w"] for p in placed.values()]
    ys = [p["y"] for p in placed.values()] + [p["y"] + p["h"] for p in placed.values()]
    detour = {1: 0, -1: 0}
    left_edge = min(p["x"] for p in placed.values())
    right_edge = max(p["x"] + p["w"] for p in placed.values())
    labels = []
    for e in edges:
        a = placed.get(e["from"])
        b = placed.get(e["to"])
        if not a or not b:
            continue
        (x1, y1), (x2, y2) = anchors(a, b)
        span = abs(a["row"] - b["row"])
        if span == 0:
            dip = a["y"] + a["h"] + 34 + same_row_seen[a["row"]] * 22
            same_row_seen[a["row"]] += 1
            cx, cy = (x1 + x2) / 2, dip
        elif span == 1:
            spread = 22 if x1 != x2 else 0
            cx = (x1 + x2) / 2 + spread
            cy = (y1 + y2) / 2
        else:
            side = 1 if (x2 >= x1 and right_edge - max(x1, x2) > max(x1, x2) - left_edge) or x2 > x1 else -1
            reach = 220 + 60 * span + detour[side] * 56
            detour[side] += 1
            cx = (x1 + x2) / 2 + side * reach
            cy = (y1 + y2) / 2
        parts.append(
            f'<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}" fill="none" '
            f'stroke="{STROKE}" stroke-width="1.9" stroke-linecap="round" marker-end="url(#arw)"/>')
        mx = 0.25 * x1 + 0.5 * cx + 0.25 * x2
        my = 0.25 * y1 + 0.5 * cy + 0.25 * y2
        xs += [mx, x1, x2]
        ys += [my, y1, y2]
        labels.append((mx, my, e.get("label") or ""))

    for pid, p in placed.items():
        n = p["node"]
        fill = FILL.get(n.get("kind"), "#f1efe9")
        parts.append(
            f'<rect x="{p["x"]:.1f}" y="{p["y"]:.1f}" width="{p["w"]}" height="{p["h"]}" rx="9" '
            f'fill="{fill}" stroke="{STROKE}" stroke-width="2"/>')
    parts.append('</g>')

    for p in placed.values():
        n = p["node"]
        label = html.escape(n["label"])
        size = 19 if len(label) <= 22 else 16
        parts.append(
            f'<text x="{p["x"] + p["w"] / 2:.1f}" y="{p["y"] + p["h"] / 2 + 6:.1f}" text-anchor="middle" '
            f'font-family="Caveat, Bradley Hand, Chalkboard SE, Comic Sans MS, cursive" '
            f'font-size="{size}" font-weight="700" fill="#22201c">{label}</text>')

    boxes = [(p["x"] - 4, p["y"] - 4, p["x"] + p["w"] + 4, p["y"] + p["h"] + 4)
             for p in placed.values()]
    for mx, my, text in labels:
        if not text:
            continue
        label = html.escape(text)
        wdt = len(label) * 7.6 + 12
        for _ in range(14):
            rect = (mx - wdt / 2, my - 12, mx + wdt / 2, my + 9)
            if not any(rect[0] < b[2] and rect[2] > b[0] and rect[1] < b[3] and rect[3] > b[1]
                       for b in boxes):
                break
            my += 24
        boxes.append((mx - wdt / 2 - 3, my - 14, mx + wdt / 2 + 3, my + 11))
        xs += [mx - wdt / 2, mx + wdt / 2]
        ys += [my - 14, my + 11]
        parts.append(
            f'<rect x="{mx - wdt / 2:.1f}" y="{my - 12:.1f}" width="{wdt:.1f}" height="21" rx="6" '
            'fill="#ffffff" fill-opacity="0.94"/>'
            f'<text x="{mx:.1f}" y="{my + 3:.1f}" text-anchor="middle" '
            'font-family="Caveat, Bradley Hand, Chalkboard SE, Comic Sans MS, cursive" '
            f'font-size="16" fill="#5f5a51">{label}</text>')
    parts.append('</svg>')
    pad = 26
    minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - pad, max(ys) + pad
    vw, vh = maxx - minx, maxy - miny
    svg = "\n".join(parts)
    svg = svg.replace('__SVG_OPEN__',
        f'<svg viewBox="{minx:.0f} {miny:.0f} {vw:.0f} {vh:.0f}" width="{vw:.0f}" height="{vh:.0f}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="architecture overview">')
    svg = svg.replace('__SVG_BG__',
        f'<rect x="{minx:.0f}" y="{miny:.0f}" width="{vw:.0f}" height="{vh:.0f}" fill="#ffffff"/>')
    return svg


def lines_of(text):
    return len([l for l in str(text).strip().splitlines() if l.strip()]) or 1


def sentences(text):
    return max(1, str(text).count(".") + str(text).count(";"))


def validate(analysis, facts, root):
    errors = []
    warnings = []

    def exists(path):
        return path and os.path.exists(os.path.join(root, path))

    for key in ("project", "architecture", "modules", "committers", "techDebt",
                "observability", "tests", "verification", "schema"):
        if key not in analysis:
            errors.append(f"analysis.json is missing the top level key '{key}'")
    if errors:
        raise Problem("\n".join(errors))

    node_ids = {n["id"] for n in analysis["architecture"].get("nodes", [])}
    if not node_ids:
        errors.append("architecture.nodes is empty; the diagram cannot be drawn")
    for e in analysis["architecture"].get("edges", []):
        if e["from"] not in node_ids:
            errors.append(f"architecture edge references unknown node '{e['from']}'")
        if e["to"] not in node_ids:
            errors.append(f"architecture edge references unknown node '{e['to']}'")
        if not e.get("evidence"):
            errors.append(f"architecture edge {e['from']} -> {e['to']} has no evidence")

    if not analysis["modules"]:
        errors.append("modules is empty")
    module_ids = set()
    for m in analysis["modules"]:
        module_ids.add(m["id"])
        where = f"module '{m['id']}'"
        if len(m.get("pros", [])) != 5:
            errors.append(f"{where} has {len(m.get('pros', []))} pros, exactly 5 are required")
        if len(m.get("cons", [])) != 5:
            errors.append(f"{where} has {len(m.get('cons', []))} cons, exactly 5 are required")
        dl = lines_of(m.get("description", ""))
        if not (MIN_DESC_LINES <= dl <= MAX_DESC_LINES) and sentences(m.get("description", "")) > MAX_DESC_LINES:
            warnings.append(f"{where} description is {dl} lines / {sentences(m.get('description',''))} sentences, aim for 3 to 5")
        for p in m.get("paths", []):
            if not exists(p):
                errors.append(f"{where} lists path '{p}' which does not exist")
        if not m.get("files"):
            errors.append(f"{where} lists no main files")
        for f in m.get("files", []):
            if not exists(f["path"]):
                errors.append(f"{where} lists file '{f['path']}' which does not exist")
        for c in m.get("cons", []):
            if len(c.get("why", "")) > MAX_WHY:
                warnings.append(f"{where} con '{c.get('title')}' why is longer than 2 lines")
            if len(c.get("fix", "")) > MAX_FIX:
                warnings.append(f"{where} con '{c.get('title')}' fix is longer than 3 lines")
            if not c.get("fix"):
                errors.append(f"{where} con '{c.get('title')}' has no fix")

    per_module = defaultdict(int)
    for d in analysis["techDebt"]:
        per_module[d["module"]] += 1
        if d["module"] not in module_ids:
            errors.append(f"tech debt '{d['id']}' points at unknown module '{d['module']}'")
        if not d.get("examples"):
            errors.append(f"tech debt '{d['id']}' has no examples")
        for ex in d.get("examples", []):
            if not exists(ex["path"]):
                errors.append(f"tech debt '{d['id']}' example path '{ex['path']}' does not exist")
        if not d.get("howToFix"):
            errors.append(f"tech debt '{d['id']}' has no fix")
        if d.get("severity") not in ("high", "medium", "low"):
            errors.append(f"tech debt '{d['id']}' severity must be high, medium or low")
    for mod, n in per_module.items():
        if n > 10:
            errors.append(f"module '{mod}' has {n} tech debt items, the cap is 10")

    commits = [c.get("commits", 0) for c in analysis["committers"]]
    if commits != sorted(commits, reverse=True):
        errors.append("committers must be ordered by commit count, highest first")
    fact_commits = {c["email"].lower(): c["commits"] for c in facts.get("committers", [])}
    for c in analysis["committers"]:
        real = fact_commits.get(str(c.get("email", "")).lower())
        if real is not None and real != c.get("commits"):
            errors.append(f"committer {c['name']} says {c.get('commits')} commits, the scan counted {real}")

    sch = analysis["schema"]
    if sch.get("present"):
        if len(sch.get("pros", [])) != 5 or len(sch.get("cons", [])) != 5:
            errors.append("schema needs exactly 5 pros and 5 cons")
        for t in sch.get("tables", []):
            if not exists(t.get("source", "")):
                errors.append(f"schema table '{t.get('name')}' source '{t.get('source')}' does not exist")
        for q in sch.get("worstQueries", []):
            if not exists(q.get("path", "")):
                errors.append(f"worst query path '{q.get('path')}' does not exist")

    tests = analysis["tests"]
    if len(tests.get("issues", [])) > 10:
        errors.append("tests.issues is capped at 10 items")
    for i in tests.get("issues", []):
        if i.get("path") and not exists(i["path"]):
            errors.append(f"test issue '{i.get('title')}' path '{i['path']}' does not exist")
    fact_tests = facts.get("tests", {})
    for t in tests.get("types", []):
        real = fact_tests.get(t["type"])
        if real and real["files"] != t.get("files"):
            errors.append(f"test type '{t['type']}' says {t.get('files')} files, the scan counted {real['files']}")
        if real and real["cases"] != t.get("cases"):
            errors.append(f"test type '{t['type']}' says {t.get('cases')} cases, the scan counted {real['cases']}")

    obs = analysis["observability"]
    for score in (obs.get("logsScore"), obs.get("defensive", {}).get("score")):
        if not isinstance(score, int) or not 0 <= score <= 100:
            errors.append("observability scores must be integers between 0 and 100")

    if analysis["verification"].get("passes") != 3:
        errors.append("verification.passes must be 3; run all three passes")

    if errors:
        raise Problem("\n".join(f"  - {e}" for e in errors))
    return warnings


def main():
    if len(sys.argv) < 4:
        print("usage: render.py <facts.json> <analysis.json> <out-dir> [template.html]")
        sys.exit(2)
    facts_path, analysis_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    template_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "assets", "template.html")

    with open(facts_path, encoding="utf-8") as fh:
        facts = json.load(fh)
    with open(analysis_path, encoding="utf-8") as fh:
        analysis = json.load(fh)
    root = facts["repoRoot"]
    out_dir = os.path.abspath(out_dir)

    try:
        warnings = validate(analysis, facts, root)
    except Problem as err:
        print("REPORT REJECTED. Fix analysis.json and run render.py again.\n", file=sys.stderr)
        print(err, file=sys.stderr)
        sys.exit(1)

    analysis["project"].setdefault("codebaseUrl", root)
    analysis["project"]["codebaseUrl"] = root
    analysis["project"].setdefault("githubUrl", facts.get("githubUrl"))
    for c in analysis["committers"]:
        if c.get("login"):
            c["avatarUrl"] = f"https://github.com/{c['login']}.png?size=104"
        elif c.get("email"):
            digest = hashlib.md5(str(c["email"]).strip().lower().encode()).hexdigest()
            c["avatarUrl"] = f"https://www.gravatar.com/avatar/{digest}?s=104&d=404"
        else:
            c["avatarUrl"] = None

    payload = dict(analysis)
    payload["reportDir"] = out_dir
    payload["facts"] = {
        "totals": facts["totals"],
        "branch": facts.get("branch"),
        "headCommit": facts.get("headCommit"),
        "scannedAt": facts.get("scannedAt"),
        "languages": facts.get("languages", []),
        "modules": [{"dir": m["dir"], "files": m["files"], "lines": m["lines"], "kind": m["kind"]}
                    for m in facts.get("modules", [])],
        "observability": {
            "logCounts": facts["observability"]["logCounts"],
            "logDensityPerKLoc": facts["observability"]["logDensityPerKLoc"],
        },
        "tests": facts.get("tests", {}),
    }

    with open(template_path, encoding="utf-8") as fh:
        template = fh.read()
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out = (template
           .replace("__TITLE__", html.escape(analysis["project"]["name"] + " — codebase inventory"))
           .replace("__DIAGRAM__", diagram(analysis["architecture"]))
           .replace("__DATA__", data_json))

    os.makedirs(out_dir, exist_ok=True)
    index = os.path.join(out_dir, "index.html")
    with open(index, "w", encoding="utf-8") as fh:
        fh.write(out)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)

    for w in warnings:
        print(f"warning: {w}")
    print(f"report written to {index}")


if __name__ == "__main__":
    main()
