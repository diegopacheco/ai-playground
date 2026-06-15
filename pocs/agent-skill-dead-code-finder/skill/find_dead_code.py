import json
import os
import re
import sys

KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "new", "else", "do", "try", "synchronized", "assert", "throw", "super", "this"}

ENTRY_METHOD_ANNOS = {
    "GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "PatchMapping", "RequestMapping",
    "Bean", "PostConstruct", "PreDestroy", "EventListener", "Scheduled", "MessageMapping",
    "KafkaListener", "RabbitListener", "JmsListener", "ExceptionHandler", "InitBinder", "ModelAttribute",
    "Test", "ParameterizedTest", "RepeatedTest", "BeforeEach", "AfterEach", "BeforeAll", "AfterAll",
}
FRAMEWORK_INTERFACES = {
    "CommandLineRunner", "ApplicationRunner", "Runnable", "Callable", "Filter", "HandlerInterceptor",
    "WebMvcConfigurer", "ApplicationListener", "InitializingBean", "DisposableBean", "Converter",
}
FRAMEWORK_CALLBACKS = {
    "run", "call", "afterPropertiesSet", "destroy", "doFilter", "onApplicationEvent",
    "preHandle", "postHandle", "afterCompletion", "convert",
}

DECL_RE = re.compile(
    r'(?:^|\n)[ \t]*(?:@\w+(?:\([^)]*\))?[ \t\n]*)*'
    r'(?:(?:public|private|protected|static|final|abstract|default|synchronized)[ \t]+)*'
    r'(?:<[^>]+>[ \t]+)?'
    r'([\w.<>\[\], ?]+?)[ \t]+(\w+)[ \t]*\(((?:[^()]|\([^()]*\))*)\)'
    r'(?:[ \t]*throws[ \t]+[\w., \t]+)?[ \t]*\{'
)
CLASS_DECL_RE = re.compile(r'\b(class|interface|enum|record)\s+(\w+)')
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


def simple_type(t):
    return t.strip().split('.')[-1].split('<')[0].strip()


def parse_classes(text, rel, index):
    out = []
    for m in CLASS_DECL_RE.finditer(text):
        name = m.group(2)
        brace = text.find('{', m.end())
        decl = re.sub(r'<[^<>]*>', '', text[m.end():brace if brace >= 0 else len(text)])
        supers = set()
        ext = re.search(r'\bextends\s+([^{]*?)(?:\bimplements\b|$)', decl)
        impl = re.search(r'\bimplements\s+([^{]*?)$', decl)
        for clause in (ext, impl):
            if clause:
                for tok in clause.group(1).split(','):
                    s = simple_type(tok)
                    if s:
                        supers.add(s)
        boundary = max(text.rfind(';', 0, m.start()), text.rfind('}', 0, m.start()))
        pre = text[boundary + 1:m.start()] if boundary >= 0 else text[:m.start()]
        annos = set(re.findall(r'@(\w+)', pre))
        entry = index.setdefault(name, {"annos": set(), "supers": set(), "methods": set(), "file": rel})
        entry["annos"].update(annos)
        entry["supers"].update(supers)
        out.append((m.start(), name))
    return out


def parse_file(path, rel, index):
    raw = open(path, encoding="utf-8").read()
    text = strip_noise(raw)
    classes = parse_classes(text, rel, index)
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
        head = text[m.start():m.start(2)]
        annos = set(re.findall(r'@(\w+)', head))
        is_static = bool(re.search(r'\bstatic\b', head))
        is_public = bool(re.search(r'\bpublic\b', head))
        params = re.sub(r'@\w+(?:\([^)]*\))?\s*', '', m.group(3)).strip()
        for pm in re.finditer(r'([A-Z]\w*)(?:<[^,)]*>)?\s+(\w+)', params):
            fields[pm.group(2)] = pm.group(1)
        line = text.count('\n', 0, m.start(2)) + 1
        body = body_of(text, m.end() - 1)
        index.setdefault(cls, {"annos": set(), "supers": set(), "methods": set(), "file": rel})["methods"].add(name)
        methods.append({
            "cls": cls, "name": name, "sig": re.sub(r'\s+', ' ', params),
            "file": rel, "line": line, "body": body, "fields": dict(fields),
            "annos": annos, "static": is_static, "public": is_public, "params": params,
        })
    return methods


def closure(start, edges, memo):
    if start in memo:
        return memo[start]
    seen = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in edges.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    memo[start] = seen
    return seen


def resolve_calls(methods, index):
    by_id = {m["cls"] + "." + m["name"]: m for m in methods}
    declares = {}
    for m in methods:
        declares.setdefault(m["cls"], set()).add(m["name"])
    up = {c: set(meta["supers"]) for c, meta in index.items()}
    down = {}
    for c, sups in up.items():
        for s in sups:
            down.setdefault(s, set()).add(c)
    up_memo, down_memo = {}, {}

    def candidates(type_name, method_name):
        types = {type_name} | closure(type_name, up, up_memo) | closure(type_name, down, down_memo)
        return [t + "." + method_name for t in types if method_name in declares.get(t, ())]

    for m in methods:
        vars_ = dict(m["fields"])
        for lm in LOCAL_RE.finditer(m["body"]):
            vars_[lm.group(2)] = lm.group(1)
        mid = m["cls"] + "." + m["name"]
        calls = []

        def add(target):
            if target and target != mid and target not in calls and target in by_id:
                calls.append(target)

        for cm in QUAL_CALL_RE.finditer(m["body"]):
            owner, name = cm.group(1), cm.group(2)
            if owner == "this":
                owner_type = m["cls"]
            elif owner in vars_:
                owner_type = vars_[owner]
            elif owner in declares or owner in index:
                owner_type = owner
            else:
                continue
            for t in candidates(owner_type, name):
                add(t)
        qualified = {(cm.group(1), cm.group(2)) for cm in QUAL_CALL_RE.finditer(m["body"])}
        for cm in BARE_CALL_RE.finditer(m["body"]):
            name = cm.group(1)
            if name in KEYWORDS or (name[0].isupper() and name not in declares):
                continue
            if any(q[1] == name for q in qualified) and m["body"][max(0, cm.start(1) - 1)] == '.':
                continue
            for t in candidates(m["cls"], name):
                add(t)
        m["calls"] = calls
    return by_id


def entry_kind(m, index):
    if m["name"] == "main" and m["static"]:
        return "main()"
    hit = m["annos"] & ENTRY_METHOD_ANNOS
    if hit:
        return "@" + sorted(hit)[0]
    meta = index.get(m["cls"], {})
    if m["public"] and (meta.get("supers", set()) & FRAMEWORK_INTERFACES) and m["name"] in FRAMEWORK_CALLBACKS:
        return "framework callback"
    return None


def build(src_dir):
    index = {}
    methods = []
    for root, _, files in os.walk(src_dir):
        for f in sorted(files):
            if f.endswith(".java"):
                p = os.path.join(root, f)
                methods.extend(parse_file(p, os.path.relpath(p, src_dir), index))
    by_id = resolve_calls(methods, index)
    callers = {}
    for mid, m in by_id.items():
        for c in m["calls"]:
            callers.setdefault(c, []).append(mid)

    entries = {}
    for mid, m in by_id.items():
        kind = entry_kind(m, index)
        if kind:
            entries[mid] = kind

    reachable = set(entries)
    stack = list(entries)
    while stack:
        cur = stack.pop()
        for c in by_id[cur]["calls"]:
            if c not in reachable:
                reachable.add(c)
                stack.append(c)

    dead = []
    for mid, m in sorted(by_id.items()):
        if mid in reachable:
            continue
        cs = sorted(callers.get(mid, []))
        reason = "no references" if not cs else "only dead callers"
        dead.append({
            "id": mid, "cls": m["cls"], "name": m["name"], "sig": m["sig"],
            "file": m["file"], "line": m["line"], "reason": reason, "callers": cs,
        })

    classes = {}
    for m in by_id.values():
        classes.setdefault(m["cls"], []).append(m["cls"] + "." + m["name"])
    dead_classes = sorted(c for c, ids in classes.items() if ids and all(i not in reachable for i in ids))

    entry_list = [{
        "id": mid, "cls": by_id[mid]["cls"], "name": by_id[mid]["name"], "sig": by_id[mid]["sig"],
        "file": by_id[mid]["file"], "line": by_id[mid]["line"], "kind": kind,
    } for mid, kind in sorted(entries.items())]

    total = len(by_id)
    stats = {
        "classes": len(classes), "methods": total, "entries": len(entries),
        "reachable": len(reachable), "dead": len(dead),
        "deadPct": round(100 * len(dead) / total, 1) if total else 0.0,
        "deadClasses": len(dead_classes),
    }
    return {"stats": stats, "entries": entry_list, "dead": dead, "deadClassNames": dead_classes}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — Dead Code</title>
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Patrick+Hand&display=swap" rel="stylesheet">
<style>
:root{--paper:#fbfaf6;--ink:#33312e;--soft:#8a857c;--line:#e7e1d6;--accent:#d6492f;--live:#4f9d69;--card:#fff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--paper);color:var(--ink);font-family:"Patrick Hand","Segoe UI",sans-serif;font-size:17px;padding-bottom:60px}
header{padding:22px 32px;border-bottom:2px solid var(--line)}
header h1{font-family:Caveat,cursive;font-size:42px;font-weight:700}
header h1 span{color:var(--accent)}
header p{color:var(--soft);margin-top:2px}
.wrap{max-width:1120px;margin:0 auto;padding:24px 28px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:26px}
.card{background:var(--card);border:2px solid var(--ink);border-radius:14px;padding:14px 16px;box-shadow:4px 4px 0 rgba(51,49,46,.1);transform:rotate(-.6deg)}
.card:nth-child(even){transform:rotate(.7deg)}
.card .n{font-family:Caveat,cursive;font-size:40px;font-weight:700;line-height:1}
.card .l{color:var(--soft);font-size:15px;margin-top:4px}
.card.dead .n{color:var(--accent)}
.card.live .n{color:var(--live)}
.viz{display:flex;align-items:center;gap:30px;flex-wrap:wrap;background:var(--card);border:2px solid var(--ink);border-radius:14px;padding:20px 24px;margin-bottom:28px;box-shadow:4px 4px 0 rgba(51,49,46,.1)}
.viz .legend{font-size:16px}
.viz .legend div{display:flex;align-items:center;gap:8px;margin:6px 0}
.viz .legend i{width:16px;height:16px;border-radius:5px;border:1.6px solid var(--ink);display:inline-block}
.tabs{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.tab{font:inherit;font-size:18px;border:2px solid var(--ink);background:var(--card);padding:6px 16px;border-radius:10px;cursor:pointer;box-shadow:3px 3px 0 rgba(51,49,46,.12)}
.tab.active{background:#fbe3d9;color:#a83417}
input#q{width:100%;padding:9px 14px;font:inherit;border:2px solid var(--line);border-radius:10px;background:#fff;outline:none;margin-bottom:16px}
input#q:focus{border-color:var(--accent)}
.group{margin-bottom:22px}
.group h3{font-family:Caveat,cursive;font-size:24px;border-bottom:2px dashed var(--line);padding-bottom:3px;margin-bottom:10px}
.row{background:var(--card);border:2px solid var(--ink);border-radius:12px;padding:11px 15px;margin-bottom:9px;box-shadow:3px 3px 0 rgba(51,49,46,.08)}
.row .sig{font-size:18px}
.row .sig b{color:var(--ink)}
.row .meta{color:var(--soft);font-size:15px;margin-top:3px}
.row .callers{font-size:14px;color:var(--soft);margin-top:4px}
.tag{font-size:13px;border-radius:20px;padding:2px 11px;border:1.6px solid var(--ink);margin-left:8px;white-space:nowrap}
.tag.no_references{background:#fad9d2}
.tag.only_dead_callers{background:#fbe7c8}
.tag.kind{background:#d7ecdd}
.hidden{display:none}
.empty{color:var(--soft);font-family:Caveat,cursive;font-size:28px;text-align:center;padding:50px}
footer{max-width:1120px;margin:30px auto 0;padding:16px 28px;color:var(--soft);font-size:15px;border-top:2px dashed var(--line)}
</style>
</head>
<body>
<header>
  <h1>__TITLE__ <span>dead code</span></h1>
  <p>static reachability from entry points</p>
</header>
<div class="wrap">
  <div class="cards" id="cards"></div>
  <div class="viz">
    <svg id="donut" width="190" height="190" viewBox="0 0 190 190"></svg>
    <div class="legend" id="legend"></div>
  </div>
  <div class="tabs">
    <button class="tab active" data-t="dead">dead code</button>
    <button class="tab" data-t="entries">entry points</button>
  </div>
  <input id="q" type="search" placeholder="filter by class, method, file...">
  <div id="dead"></div>
  <div id="entries" class="hidden"></div>
</div>
<footer id="foot"></footer>
<script>
const D = __DATA__;
const S = D.stats;

function hash(s){let h=2166136261;for(const c of s){h^=c.charCodeAt(0);h=Math.imul(h,16777619);}return h>>>0;}
function rng(seed){let s=seed||1;return()=>{s=Math.imul(s^(s>>>15),s|1);s^=s+Math.imul(s^(s>>>7),s|61);return((s^(s>>>14))>>>0)/4294967296;};}

function cards(){
  const live=S.reachable, items=[
    ["methods",S.methods,""],["classes",S.classes,""],["entry points",S.entries,""],
    ["reachable",live,"live"],["dead methods",S.dead,"dead"],["dead %",S.deadPct+"%","dead"],["dead classes",S.deadClasses,"dead"]
  ];
  document.getElementById("cards").innerHTML=items.map(([l,n,c])=>
    `<div class="card ${c}"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
}

function arc(cx,cy,r,a0,a1,seed){
  const r2=rng(seed);const j=()=>(r2()*2-1)*1.4;
  const p0=[cx+r*Math.cos(a0),cy+r*Math.sin(a0)];
  const p1=[cx+r*Math.cos(a1),cy+r*Math.sin(a1)];
  const big=a1-a0>Math.PI?1:0;
  return `M${cx+j()} ${cy+j()} L${p0[0]+j()} ${p0[1]+j()} A${r} ${r} 0 ${big} 1 ${p1[0]+j()} ${p1[1]+j()} Z`;
}
function donut(){
  const dead=S.dead, live=S.reachable, tot=dead+live||1;
  const cx=95,cy=95,r=78,start=-Math.PI/2;
  const split=start+2*Math.PI*live/tot;
  let svg="";
  svg+=`<path d="${arc(cx,cy,r,start,split,7)}" fill="#cdeacf" stroke="#33312e" stroke-width="2.2" stroke-linejoin="round"/>`;
  svg+=`<path d="${arc(cx,cy,r,split,start+2*Math.PI,21)}" fill="#f6cabf" stroke="#33312e" stroke-width="2.2" stroke-linejoin="round"/>`;
  svg+=`<circle cx="${cx}" cy="${cy}" r="40" fill="#fbfaf6" stroke="#33312e" stroke-width="2.2"/>`;
  svg+=`<text x="${cx}" y="${cy-4}" text-anchor="middle" font-family="Caveat,cursive" font-size="34" font-weight="700" fill="#d6492f">${S.deadPct}%</text>`;
  svg+=`<text x="${cx}" y="${cy+18}" text-anchor="middle" font-family="Patrick Hand" font-size="15" fill="#8a857c">dead</text>`;
  document.getElementById("donut").innerHTML=svg;
  document.getElementById("legend").innerHTML=
    `<div><i style="background:#cdeacf"></i> ${live} reachable methods</div>`+
    `<div><i style="background:#f6cabf"></i> ${S.dead} dead methods</div>`+
    `<div><i style="background:#fff"></i> ${S.entries} entry points seed the analysis</div>`;
}

function deadRows(filter){
  const f=(filter||"").toLowerCase();
  const list=D.dead.filter(d=>!f||d.id.toLowerCase().includes(f)||d.file.toLowerCase().includes(f));
  const box=document.getElementById("dead");
  if(!list.length){box.innerHTML='<div class="empty">no dead code — every method is reachable 🎉</div>';return;}
  const byFile={};
  for(const d of list)(byFile[d.file]=byFile[d.file]||[]).push(d);
  box.innerHTML=Object.keys(byFile).sort().map(file=>{
    const rows=byFile[file].sort((a,b)=>a.line-b.line).map(d=>{
      const dc=D.deadClassNames.includes(d.cls)?' · <b style="color:#d6492f">dead class</b>':'';
      const callers=d.callers.length?`<div class="callers">called by ${d.callers.map(c=>c+"()").join(", ")}</div>`:'';
      return `<div class="row"><div class="sig"><b>${d.cls}.${d.name}</b>(${d.sig})<span class="tag ${d.reason.replace(/ /g,"_")}">${d.reason}</span></div>`+
        `<div class="meta">${d.file}:${d.line}${dc}</div>${callers}</div>`;
    }).join("");
    return `<div class="group"><h3>${file}</h3>${rows}</div>`;
  }).join("");
}

function entryRows(filter){
  const f=(filter||"").toLowerCase();
  const list=D.entries.filter(e=>!f||e.id.toLowerCase().includes(f)||e.file.toLowerCase().includes(f));
  const box=document.getElementById("entries");
  if(!list.length){box.innerHTML='<div class="empty">no entry points found</div>';return;}
  const byFile={};
  for(const e of list)(byFile[e.file]=byFile[e.file]||[]).push(e);
  box.innerHTML=Object.keys(byFile).sort().map(file=>{
    const rows=byFile[file].sort((a,b)=>a.line-b.line).map(e=>
      `<div class="row"><div class="sig"><b>${e.cls}.${e.name}</b>(${e.sig})<span class="tag kind">${e.kind}</span></div>`+
      `<div class="meta">${e.file}:${e.line}</div></div>`).join("");
    return `<div class="group"><h3>${file}</h3>${rows}</div>`;
  }).join("");
}

let tab="dead";
function refresh(){const q=document.getElementById("q").value;deadRows(q);entryRows(q);}
document.querySelectorAll(".tab").forEach(b=>b.addEventListener("click",()=>{
  tab=b.dataset.t;
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x===b));
  document.getElementById("dead").classList.toggle("hidden",tab!=="dead");
  document.getElementById("entries").classList.toggle("hidden",tab!=="entries");
}));
document.getElementById("q").addEventListener("input",refresh);
document.getElementById("foot").textContent=
  "Heuristics: entry points are main(), Spring web/lifecycle/scheduler annotations, JUnit tests, and framework callbacks. "+
  "Calls are resolved from source by declared type plus implements/extends, so reflection, dynamic proxies and runtime configuration are not seen — review flagged items before deleting.";
cards();donut();refresh();
</script>
</body>
</html>
"""


def main():
    if len(sys.argv) < 3:
        print("usage: find_dead_code.py <java-src-dir> <out-dir> [title]")
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.abspath(src.rstrip("/")))
    data = build(src)
    if not data["stats"]["methods"]:
        print("no java methods found under " + src)
        sys.exit(1)
    os.makedirs(out, exist_ok=True)
    page = PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    s = data["stats"]
    print("dead-code site written to %s/index.html" % out)
    print("%d methods, %d entry points, %d reachable, %d dead (%.1f%%), %d dead classes"
          % (s["methods"], s["entries"], s["reachable"], s["dead"], s["deadPct"], s["deadClasses"]))


if __name__ == "__main__":
    main()
