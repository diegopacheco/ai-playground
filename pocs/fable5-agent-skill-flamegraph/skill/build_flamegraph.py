import json
import os
import re
import sys

KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "new", "else", "do", "try", "synchronized", "assert", "throw", "super", "this"}

DECL_RE = re.compile(
    r'(?:^|\n)[ \t]*(?:@\w+(?:\([^)]*\))?[ \t\n]*)*'
    r'(?:(?:public|private|protected|static|final|abstract|default|synchronized)[ \t]+)*'
    r'(?:<[^>]+>[ \t]+)?'
    r'([\w.<>\[\], ?]+?)[ \t]+(\w+)[ \t]*\(((?:[^()]|\([^()]*\))*)\)'
    r'(?:[ \t]*throws[ \t]+[\w., \t]+)?[ \t]*\{'
)

CLASS_RE = re.compile(r'\b(?:class|interface|record|enum)\s+(\w+)')
FIELD_RE = re.compile(r'\b(?:private|protected|public)\s+(?:final\s+|static\s+)*([A-Z]\w*)(?:<[^;=]*>)?\s+(\w+)\s*[;=]')
LOCAL_RE = re.compile(r'\b([A-Z]\w*)(?:<[^;=()]*>)?\s+(\w+)\s*=')
QUAL_CALL_RE = re.compile(r'\b(\w+)\s*\.\s*(\w+)\s*\(')
BARE_CALL_RE = re.compile(r'(?<![\w.])(\w+)\s*\(')


def strip_noise(text):
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def body_of(text, open_brace):
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    return text[open_brace + 1:]


def parse_file(path, rel):
    raw = open(path, encoding="utf-8").read()
    text = strip_noise(raw)
    classes = [(m.start(), m.group(1)) for m in CLASS_RE.finditer(text)]
    if not classes:
        return []
    fields = {}
    for m in FIELD_RE.finditer(text):
        fields[m.group(2)] = m.group(1)
    methods = []
    for m in DECL_RE.finditer(text):
        name = m.group(2)
        parts = m.group(1).strip().split()
        ret = parts[-1] if parts else ""
        if name in KEYWORDS or ret in KEYWORDS or ret in {"record", "class", "interface", "enum"}:
            continue
        cls = classes[0][1]
        for pos, cname in classes:
            if pos < m.start(3):
                cls = cname
        if name == cls:
            continue
        params = re.sub(r'@\w+(?:\([^)]*\))?\s*', '', m.group(3)).strip()
        for pm in re.finditer(r'([A-Z]\w*)(?:<[^,)]*>)?\s+(\w+)', params):
            fields[pm.group(2)] = pm.group(1)
        line = text.count('\n', 0, m.start(2)) + 1
        body = body_of(text, m.end() - 1)
        methods.append({
            "cls": cls, "name": name, "sig": re.sub(r'\s+', ' ', params),
            "file": rel, "line": line, "body": body, "fields": dict(fields),
        })
    return methods


def resolve_calls(methods):
    by_id = {m["cls"] + "." + m["name"]: m for m in methods}
    by_class = {}
    for m in methods:
        by_class.setdefault(m["cls"], set()).add(m["name"])
    for m in methods:
        vars_ = dict(m["fields"])
        for lm in LOCAL_RE.finditer(m["body"]):
            vars_[lm.group(2)] = lm.group(1)
        calls = []
        for cm in QUAL_CALL_RE.finditer(m["body"]):
            owner, name = cm.group(1), cm.group(2)
            target = None
            if owner in vars_ and name in by_class.get(vars_[owner], ()):
                target = vars_[owner] + "." + name
            elif owner in by_class and name in by_class[owner]:
                target = owner + "." + name
            elif owner == "this" and name in by_class.get(m["cls"], ()):
                target = m["cls"] + "." + name
            if target and target not in calls:
                calls.append(target)
        qualified = {(cm.group(1), cm.group(2)) for cm in QUAL_CALL_RE.finditer(m["body"])}
        for cm in BARE_CALL_RE.finditer(m["body"]):
            name = cm.group(1)
            if name in KEYWORDS or (name[0].isupper() and name not in by_class):
                continue
            if any(q[1] == name and cm.start(1) - 1 >= 0 for q in qualified) and m["body"][max(0, cm.start(1) - 1)] == '.':
                continue
            if name in by_class.get(m["cls"], ()):
                target = m["cls"] + "." + name
                if target != m["cls"] + "." + m["name"] and target not in calls:
                    calls.append(target)
        m["calls"] = calls
    return by_id


def build_graph(src_dir):
    methods = []
    for root, _, files in os.walk(src_dir):
        for f in sorted(files):
            if f.endswith(".java"):
                p = os.path.join(root, f)
                methods.extend(parse_file(p, os.path.relpath(p, src_dir)))
    by_id = resolve_calls(methods)
    callers = {}
    for mid, m in by_id.items():
        for c in m["calls"]:
            callers.setdefault(c, []).append(mid)
    out = {}
    for mid, m in by_id.items():
        out[mid] = {
            "cls": m["cls"], "name": m["name"], "sig": m["sig"],
            "file": m["file"], "line": m["line"],
            "calls": m["calls"], "callers": sorted(callers.get(mid, [])),
        }
    return out


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — Call-Graph Flamegraph</title>
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Patrick+Hand&display=swap" rel="stylesheet">
<style>
:root{--paper:#fdfbf6;--ink:#33312e;--soft:#8a857c;--line:#e7e1d6;--accent:#e8633a}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--paper);color:var(--ink);font-family:"Patrick Hand","Segoe UI",sans-serif;font-size:17px}
header{display:flex;align-items:baseline;gap:16px;padding:18px 28px;border-bottom:2px solid var(--line)}
header h1{font-family:Caveat,cursive;font-size:38px;font-weight:700}
header h1 span{color:var(--accent)}
header .stats{color:var(--soft);font-size:16px}
.layout{display:grid;grid-template-columns:320px 1fr;height:calc(100vh - 79px)}
aside{border-right:2px solid var(--line);overflow-y:auto;padding:14px}
aside input{width:100%;padding:8px 12px;font:inherit;border:2px solid var(--line);border-radius:10px;background:#fff;outline:none;margin-bottom:12px}
aside input:focus{border-color:var(--accent)}
.cls{font-family:Caveat,cursive;font-size:22px;font-weight:700;margin:10px 4px 2px;display:flex;align-items:center;gap:8px}
.cls i{display:inline-block;width:12px;height:12px;border-radius:4px;transform:rotate(-3deg)}
.mth{display:block;width:100%;text-align:left;border:none;background:none;font:inherit;color:var(--ink);padding:3px 10px 3px 26px;border-radius:8px;cursor:pointer}
.mth:hover{background:#f3ede1}
.mth.active{background:#fbe3d9;color:#b3431f}
main{overflow-y:auto;padding:20px 28px}
.hint{margin-top:60px;text-align:center;color:var(--soft);font-family:Caveat,cursive;font-size:30px}
.crumbs{margin-bottom:6px;color:var(--soft)}
.crumbs button{border:none;background:none;font:inherit;color:var(--accent);cursor:pointer;text-decoration:underline}
#chart{width:100%}
#detail{margin-top:18px;border-top:2px dashed var(--line);padding-top:12px;display:none}
#detail h2{font-family:Caveat,cursive;font-size:28px}
#detail .loc{color:var(--soft)}
#detail .cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px}
#detail h3{font-family:Caveat,cursive;font-size:22px;color:var(--accent)}
#detail li{list-style:none}
#detail li button{border:none;background:none;font:inherit;color:var(--ink);cursor:pointer;text-decoration:underline dotted}
#tip{position:fixed;pointer-events:none;background:#fffdf7;border:2px solid var(--ink);border-radius:10px;padding:6px 12px;font-size:15px;display:none;box-shadow:3px 3px 0 rgba(51,49,46,.15);max-width:340px;z-index:9}
#tip b{font-family:Caveat,cursive;font-size:19px}
text{font-family:Caveat,cursive;font-weight:600;pointer-events:none}
.frame{cursor:pointer}
.frame:hover path{filter:brightness(.96)}
</style>
</head>
<body>
<header>
  <h1>__TITLE__ <span>flamegraph</span></h1>
  <div class="stats" id="stats"></div>
</header>
<div class="layout">
  <aside>
    <input id="search" type="search" placeholder="find a class or method...">
    <div id="list"></div>
  </aside>
  <main>
    <div class="crumbs" id="crumbs"></div>
    <div id="chart"><div class="hint">pick a class or method on the left &mdash; its call tree blooms here as a flamegraph</div></div>
    <div id="detail"></div>
  </main>
</div>
<div id="tip"></div>
<script>
const GRAPH = __DATA__;
const PALETTE = ["#f9c8c0","#fbe2b8","#fdf3b5","#d7ecc0","#bfe5d8","#c2dff5","#cfcdf2","#ecccec","#f5d5c2","#d9e8b8","#bfe9ea","#e3d4f4"];
const FRAME_H = 36, GAP = 5;
let stack = [];

const classes = [...new Set(Object.values(GRAPH).map(m => m.cls))].sort();
const classColor = c => PALETTE[classes.indexOf(c) % PALETTE.length];

function hash(s){let h = 2166136261; for (const ch of s){h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619);} return h >>> 0;}
function rng(seed){let s = seed || 1; return () => {s = Math.imul(s ^ (s >>> 15), s | 1); s ^= s + Math.imul(s ^ (s >>> 7), s | 61); return ((s ^ (s >>> 14)) >>> 0) / 4294967296;};}

function sketchRect(x, y, w, h, seed){
  const r = rng(seed), j = () => (r() * 2 - 1) * Math.min(2.2, w / 14);
  const pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
  let d = `M${(x + j()).toFixed(1)} ${(y + j()).toFixed(1)}`;
  for (let i = 0; i < 4; i++){
    const a = pts[i], b = pts[(i + 1) % 4];
    const cx = (a[0] + b[0]) / 2 + j() * 1.6, cy = (a[1] + b[1]) / 2 + j() * 1.6;
    d += ` Q${cx.toFixed(1)} ${cy.toFixed(1)} ${(b[0] + j()).toFixed(1)} ${(b[1] + j()).toFixed(1)}`;
  }
  return d + " Z";
}

function buildTree(id, depth, path){
  const node = {id, children: [], cycle: path.has(id)};
  if (node.cycle || depth > 13) return node;
  path.add(id);
  for (const c of GRAPH[id].calls) if (GRAPH[c]) node.children.push(buildTree(c, depth + 1, path));
  path.delete(id);
  return node;
}
function leaves(n){return n.children.length ? n.children.reduce((s, c) => s + leaves(c), 0) : 1;}
function depthOf(n){return n.children.length ? 1 + Math.max(...n.children.map(depthOf)) : 1;}

function render(rootId, push){
  if (push !== false){ if (stack[stack.length - 1] !== rootId) stack.push(rootId); }
  const tree = buildTree(rootId, 0, new Set());
  const W = document.getElementById("chart").clientWidth || 900;
  const D = depthOf(tree);
  const H = D * (FRAME_H + GAP) + 12;
  let svg = `<svg id="fg" viewBox="0 0 ${W} ${H}" width="100%" height="${H}">`;
  const total = leaves(tree);
  function draw(n, x, w, d){
    const m = GRAPH[n.id];
    const y = d * (FRAME_H + GAP) + 4;
    const label = w > 130 ? `${m.cls}.${m.name}()` : (w > 56 ? m.name : "");
    const fill = n.cycle ? "#eee9df" : classColor(m.cls);
    svg += `<g class="frame" data-id="${n.id}" data-cycle="${n.cycle ? 1 : 0}">`;
    svg += `<path d="${sketchRect(x + 1, y, Math.max(w - 2, 4), FRAME_H, hash(n.id + d + Math.round(x)))}" fill="${fill}" stroke="#4a463f" stroke-width="1.7" stroke-linejoin="round"/>`;
    if (label) svg += `<text x="${x + w / 2}" y="${y + FRAME_H / 2 + 7}" text-anchor="middle" font-size="19" fill="#3a352c">${label}${n.cycle ? " ↺" : ""}</text>`;
    svg += `</g>`;
    let cx = x;
    const lv = leaves(n);
    for (const c of n.children){
      const cw = w * leaves(c) / lv;
      draw(c, cx, cw, d + 1);
      cx += cw;
    }
  }
  draw(tree, 0, W, 0);
  svg += "</svg>";
  document.getElementById("chart").innerHTML = svg;
  document.querySelectorAll(".frame").forEach(g => {
    g.addEventListener("click", () => render(g.dataset.id));
    g.addEventListener("mousemove", e => tip(e, g.dataset.id, g.dataset.cycle === "1"));
    g.addEventListener("mouseleave", () => document.getElementById("tip").style.display = "none");
  });
  crumbs();
  detail(rootId, total, D);
  document.querySelectorAll(".mth").forEach(b => b.classList.toggle("active", b.dataset.id === rootId));
}

function tip(e, id, cycle){
  const m = GRAPH[id], t = document.getElementById("tip");
  t.innerHTML = `<b>${m.cls}.${m.name}(${m.sig})</b><br>${m.file}:${m.line}<br>calls ${m.calls.length} · called by ${m.callers.length}${cycle ? "<br>↺ recursive cycle, pruned here" : ""}`;
  t.style.display = "block";
  t.style.left = Math.min(e.clientX + 14, innerWidth - 360) + "px";
  t.style.top = (e.clientY + 16) + "px";
}

function crumbs(){
  const c = document.getElementById("crumbs");
  c.innerHTML = stack.map((id, i) =>
    i === stack.length - 1 ? `<b>${GRAPH[id].cls}.${GRAPH[id].name}</b>` : `<button data-i="${i}">${GRAPH[id].cls}.${GRAPH[id].name}</button>`
  ).join(" &rsaquo; ");
  c.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    stack = stack.slice(0, +b.dataset.i + 1);
    render(stack[stack.length - 1], false);
  }));
}

function detail(id, total, depth){
  const m = GRAPH[id], d = document.getElementById("detail");
  const li = ids => ids.length ? ids.map(x => `<li><button data-id="${x}">${x}()</button></li>`).join("") : "<li>&mdash;</li>";
  d.style.display = "block";
  d.innerHTML = `<h2>${m.cls}.${m.name}(${m.sig})</h2>
    <div class="loc">${m.file}:${m.line} &nbsp;·&nbsp; ${total} leaf path${total === 1 ? "" : "s"}, ${depth} level${depth === 1 ? "" : "s"} deep</div>
    <div class="cols"><div><h3>calls</h3><ul>${li(m.calls)}</ul></div>
    <div><h3>called by</h3><ul>${li(m.callers)}</ul></div></div>`;
  d.querySelectorAll("button").forEach(b => b.addEventListener("click", () => render(b.dataset.id)));
}

function sidebar(filter){
  const box = document.getElementById("list");
  const f = (filter || "").toLowerCase();
  box.innerHTML = classes.map(cls => {
    const ms = Object.keys(GRAPH).filter(id => GRAPH[id].cls === cls)
      .filter(id => !f || id.toLowerCase().includes(f))
      .sort((a, b) => GRAPH[a].line - GRAPH[b].line);
    if (!ms.length) return "";
    return `<div class="cls"><i style="background:${classColor(cls)};border:1.5px solid #4a463f"></i>${cls}</div>` +
      ms.map(id => `<button class="mth" data-id="${id}">${GRAPH[id].name}(${GRAPH[id].sig ? "…" : ""})</button>`).join("");
  }).join("");
  box.querySelectorAll(".mth").forEach(b => b.addEventListener("click", () => { stack = []; render(b.dataset.id); }));
}

document.getElementById("search").addEventListener("input", e => sidebar(e.target.value));
document.getElementById("stats").textContent =
  `${classes.length} classes · ${Object.keys(GRAPH).length} methods · ${Object.values(GRAPH).reduce((s, m) => s + m.calls.length, 0)} call edges`;
sidebar("");
const roots = Object.keys(GRAPH).filter(id => !GRAPH[id].callers.length && GRAPH[id].calls.length);
if (roots.length) render(roots.sort((a, b) => leaves(buildTree(b, 0, new Set())) - leaves(buildTree(a, 0, new Set())))[0]);
addEventListener("resize", () => { if (stack.length) render(stack[stack.length - 1], false); });
</script>
</body>
</html>
"""


def main():
    if len(sys.argv) < 3:
        print("usage: build_flamegraph.py <java-src-dir> <out-dir> [title]")
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.abspath(src.rstrip("/")))
    graph = build_graph(src)
    if not graph:
        print("no java methods found under " + src)
        sys.exit(1)
    os.makedirs(out, exist_ok=True)
    page = PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(graph))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    edges = sum(len(m["calls"]) for m in graph.values())
    print("flamegraph site written to %s/index.html (%d methods, %d call edges)" % (out, len(graph), edges))


if __name__ == "__main__":
    main()
