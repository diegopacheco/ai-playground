#!/usr/bin/env python3
import sys
import os
import re
import json
import socket
import subprocess
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIFF_MAX_LINES = 160
DIFF_MAX_CHARS = 6000
UNIT = "\x1f"
REC = "\x1e"


def git(repo, args, check=True):
    proc = subprocess.run(
        ["git", "-C", repo] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    if check and proc.returncode != 0:
        raise RuntimeError("git " + " ".join(args) + " failed:\n" + err)
    return proc.returncode, out, err


def is_repo(repo):
    code, out, _ = git(repo, ["rev-parse", "--is-inside-work-tree"], check=False)
    return code == 0 and out.strip() == "true"


def truncate_diff(text):
    lines = text.splitlines()
    truncated = False
    if len(lines) > DIFF_MAX_LINES:
        lines = lines[:DIFF_MAX_LINES]
        truncated = True
    text = "\n".join(lines)
    if len(text) > DIFF_MAX_CHARS:
        text = text[:DIFF_MAX_CHARS]
        truncated = True
    if truncated:
        text = text + "\n... (diff truncated)"
    return text


def collect(repo, out_path, count):
    if not is_repo(repo):
        raise RuntimeError(repo + " is not a git repository")
    fmt = UNIT.join(["%H", "%h", "%an", "%aI", "%P", "%s", "%b"]) + REC
    _, raw, _ = git(repo, ["log", "-n", str(count), "--pretty=format:" + fmt])
    commits = []
    for record in raw.split(REC):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(UNIT)
        if len(parts) < 7:
            continue
        full, short, author, date, parents, subject, body = parts[:7]
        is_merge = len(parents.split()) > 1
        _, numstat, _ = git(repo, ["show", full, "--numstat", "--format="])
        files = []
        for line in numstat.splitlines():
            cols = line.split("\t")
            if len(cols) != 3:
                continue
            add, dele, path = cols
            files.append({"path": path, "add": add, "del": dele})
        _, diff, _ = git(repo, ["show", full, "-p", "--format=", "--no-color"])
        commits.append({
            "hash": full,
            "short": short,
            "author": author,
            "date": date,
            "is_merge": is_merge,
            "old_subject": subject,
            "old_body": body.strip(),
            "files": files,
            "diff": truncate_diff(diff),
        })
    with open(out_path, "w") as fh:
        json.dump(commits, fh, indent=2)
    print("collected " + str(len(commits)) + " commits to " + out_path)
    if commits:
        print("newest: " + commits[0]["short"] + " " + commits[0]["old_subject"])
        print("oldest: " + commits[-1]["short"] + " " + commits[-1]["old_subject"])


TYPE_RE = re.compile(r"^([a-z]+)(\([^)]*\))?(!)?:")


def parse_type(message):
    m = TYPE_RE.match(message or "")
    if m:
        return m.group(1)
    return "chore"


def load_merged(commits_path, suggestions_path):
    with open(commits_path) as fh:
        commits = json.load(fh)
    suggestions = {}
    if os.path.exists(suggestions_path):
        with open(suggestions_path) as fh:
            for s in json.load(fh):
                suggestions[s["hash"]] = s
    items = []
    for c in commits:
        s = suggestions.get(c["hash"])
        suggestion = None
        if s:
            msg = s.get("message", "").strip()
            suggestion = {
                "type": s.get("type") or parse_type(msg),
                "message": msg,
                "reason": s.get("reason", ""),
            }
        item = dict(c)
        item["suggestion"] = suggestion
        items.append(item)
    return items


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def rewrite(repo, mapping):
    code, _, _ = git(repo, ["diff", "--quiet"], check=False)
    code2, _, _ = git(repo, ["diff", "--cached", "--quiet"], check=False)
    if code != 0 or code2 != 0:
        raise RuntimeError("working tree is not clean; commit or stash first")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = "backup/pre-fix-git-history-" + stamp
    git(repo, ["branch", backup, "HEAD"])
    map_path = os.path.join(repo, ".fix-git-history", "msg-map.json")
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    with open(map_path, "w") as fh:
        json.dump(mapping, fh)
    hashes = list(mapping.keys())
    _, oldest_parents, _ = git(repo, ["rev-list", "--parents", "-n", "1", hashes[-1]])
    oldest = hashes[-1]
    has_parent = len(oldest_parents.split()) > 1
    rev_range = oldest + "^..HEAD" if has_parent else "HEAD"
    script = os.path.abspath(__file__)
    env = dict(os.environ)
    env["FGH_MSG_MAP"] = map_path
    env["FILTER_BRANCH_SQUELCH_WARNING"] = "1"
    msg_filter = "python3 " + shell_quote(script) + " _msgfilter"
    proc = subprocess.run(
        ["git", "-C", repo, "filter-branch", "-f", "--msg-filter", msg_filter, "--", rev_range],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RuntimeError("rewrite failed:\n" + out)
    _, new_head, _ = git(repo, ["rev-parse", "--short", "HEAD"])
    return {
        "backup": backup,
        "rewritten": len(mapping),
        "new_head": new_head.strip(),
    }


def shell_quote(path):
    return "'" + path.replace("'", "'\\''") + "'"


def msgfilter():
    original = sys.stdin.read()
    map_path = os.environ.get("FGH_MSG_MAP")
    commit = os.environ.get("GIT_COMMIT", "")
    if map_path and os.path.exists(map_path):
        with open(map_path) as fh:
            mapping = json.load(fh)
        if commit in mapping:
            sys.stdout.write(mapping[commit])
            return
    sys.stdout.write(original)


class Handler(BaseHTTPRequestHandler):
    items = []
    repo = ""

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/data":
            payload = {"repo": self.repo, "items": self.items}
            self._send(200, json.dumps(payload), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/apply":
            self._send(404, "not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            req = json.loads(body)
            approved = req.get("approved", [])
            valid = {c["hash"] for c in self.items}
            mapping = {}
            for a in approved:
                h = a["hash"]
                msg = a["message"].strip()
                if h in valid and msg:
                    mapping[h] = msg
            if not mapping:
                raise RuntimeError("nothing approved")
            result = rewrite(self.repo, mapping)
            self._send(200, json.dumps({"ok": True, "result": result}), "application/json")
        except Exception as exc:
            self._send(200, json.dumps({"ok": False, "error": str(exc)}), "application/json")


def serve(repo, commits_path, suggestions_path, port):
    if not is_repo(repo):
        raise RuntimeError(repo + " is not a git repository")
    Handler.items = load_merged(commits_path, suggestions_path)
    Handler.repo = os.path.abspath(repo)
    if not port:
        port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:" + str(port) + "/"
    suggested = len([i for i in Handler.items if i["suggestion"]])
    print("review " + str(len(Handler.items)) + " commits, " + str(suggested) + " suggested changes")
    print("open " + url)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fix-git-history</title>
<style>
:root{
  --bg:#f6f7fb; --panel:#ffffff; --ink:#1f2430; --muted:#7a8194; --line:#e6e8ef;
  --accent:#4f7cff; --ok:#1f9d6b; --okbg:#e7f7ef; --shadow:0 1px 2px rgba(31,36,48,.06),0 8px 24px rgba(31,36,48,.06);
  --feat:#1f9d6b; --fix:#e0533d; --docs:#3b82f6; --refactor:#8b5cf6; --perf:#d97706;
  --test:#0e9aa3; --style:#db2777; --build:#6b7280; --ci:#6b7280; --chore:#6b7280;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
.head{position:sticky;top:0;z-index:5;background:rgba(246,247,251,.9);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:18px 24px}
.title{font-family:"Caveat","Comic Sans MS",cursive;font-size:34px;line-height:1;margin:0}
.sub{color:var(--muted);margin-top:4px;font-size:13px;word-break:break-all}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:14px}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:6px 12px;color:var(--muted)}
.pill b{color:var(--ink)}
button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:8px 14px}
button:hover{border-color:#c9cedb}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.btn-primary:hover{filter:brightness(.95)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
.spacer{flex:1}
.wrap{max-width:1000px;margin:0 auto;padding:22px 24px 120px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:16px 18px;margin-bottom:14px;transition:border-color .15s,opacity .15s}
.card.off{opacity:.5}
.card.on{border-color:#cfe0ff}
.crow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.hash{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--muted);background:#f1f3f8;border-radius:6px;padding:2px 7px}
.meta{color:var(--muted);font-size:12px}
.badge{font-size:11px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;color:#fff;border-radius:6px;padding:2px 8px}
.merge{font-size:11px;color:var(--muted);border:1px dashed var(--line);border-radius:6px;padding:2px 7px}
.switch{margin-left:auto;display:inline-flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;user-select:none}
.switch input{width:38px;height:22px;appearance:none;background:#d6dae6;border-radius:999px;position:relative;cursor:pointer;transition:background .15s}
.switch input:checked{background:var(--ok)}
.switch input:before{content:"";position:absolute;width:18px;height:18px;border-radius:50%;background:#fff;top:2px;left:2px;transition:left .15s}
.switch input:checked:before{left:18px}
.old{margin:12px 0 4px;color:var(--muted);font-size:12px}
.old code{font-family:ui-monospace,Menlo,monospace;background:#fbeaea;color:#9a4030;border-radius:5px;padding:1px 6px;text-decoration:line-through}
.old code.empty{background:#f1f3f8;color:var(--muted);text-decoration:none;font-style:italic}
.newrow{display:flex;gap:8px;align-items:center;margin-top:6px}
.newrow .arrow{color:var(--ok);font-weight:700}
input.msg{flex:1;font:14px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:9px;padding:9px 11px;color:var(--ink)}
input.msg:focus{outline:none;border-color:var(--accent)}
.reason{color:var(--muted);font-size:12px;margin-top:7px;font-style:italic}
.files{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px}
.chip{font-family:ui-monospace,Menlo,monospace;font-size:11px;background:#f1f3f8;border:1px solid var(--line);border-radius:6px;padding:2px 7px;color:#4a5160}
.chip .a{color:var(--feat)} .chip .d{color:var(--fix)}
.difftoggle{margin-top:10px;font-size:12px;color:var(--accent);background:none;border:none;padding:0}
pre.diff{display:none;margin:10px 0 0;max-height:340px;overflow:auto;background:#0f1320;color:#d7def0;border-radius:10px;padding:12px;font:12px/1.5 ui-monospace,Menlo,monospace}
pre.diff .p{color:#5fd08a} pre.diff .m{color:#ff8c7a} pre.diff .h{color:#8aa0ff}
.kept{color:var(--muted);font-size:13px;margin-top:8px}
.applybar{position:fixed;left:0;right:0;bottom:0;background:rgba(255,255,255,.96);border-top:1px solid var(--line);box-shadow:0 -8px 24px rgba(31,36,48,.06);padding:14px 24px;display:flex;align-items:center;gap:14px}
.applybar .count{color:var(--muted)}
.overlay{position:fixed;inset:0;background:rgba(31,36,48,.45);display:none;align-items:center;justify-content:center;z-index:20}
.modal{background:#fff;border-radius:16px;box-shadow:var(--shadow);padding:26px;max-width:520px;width:calc(100% - 48px)}
.modal h2{margin:0 0 10px;font-family:"Caveat",cursive;font-size:28px}
.modal code{background:#f1f3f8;border-radius:6px;padding:2px 7px;font-family:ui-monospace,Menlo,monospace}
.modal .row{margin:8px 0}
.ok-ico{color:var(--ok)} .err-ico{color:var(--fix)}
</style>
</head>
<body>
<div class="head">
  <h1 class="title">fix-git-history</h1>
  <div class="sub" id="repo"></div>
  <div class="bar">
    <span class="pill"><b id="c-total">0</b> commits</span>
    <span class="pill"><b id="c-sugg">0</b> suggested</span>
    <span class="pill"><b id="c-appr">0</b> approved</span>
    <span class="spacer"></span>
    <button id="all">Approve all</button>
    <button id="none">Reject all</button>
  </div>
</div>
<div class="wrap" id="list"></div>
<div class="applybar">
  <span class="count" id="applycount">0 commits will be rewritten</span>
  <span class="spacer"></span>
  <button class="btn-primary" id="apply" disabled>Apply approved</button>
</div>
<div class="overlay" id="overlay"><div class="modal" id="modal"></div></div>
<script>
let items=[];
const esc=s=>(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
function colorDiff(t){
  return esc(t).split("\n").map(l=>{
    if(l.startsWith("+")&&!l.startsWith("+++"))return '<span class="p">'+l+'</span>';
    if(l.startsWith("-")&&!l.startsWith("---"))return '<span class="m">'+l+'</span>';
    if(l.startsWith("@@")||l.startsWith("diff "))return '<span class="h">'+l+'</span>';
    return l;
  }).join("\n");
}
function render(){
  const list=document.getElementById("list");
  list.innerHTML="";
  items.forEach((it,i)=>{
    const card=document.createElement("div");
    card.className="card "+(it.suggestion?(it.on?"on":"off"):"");
    const t=it.suggestion?it.suggestion.type:null;
    const badge=t?'<span class="badge" style="background:var(--'+t+',#6b7280)">'+esc(t)+'</span>':'';
    const merge=it.is_merge?'<span class="merge">merge</span>':'';
    const oldMsg=it.old_subject?'<code>'+esc(it.old_subject)+'</code>':'<code class="empty">(empty message)</code>';
    const files=(it.files||[]).map(f=>'<span class="chip">'+esc(f.path)+' <span class="a">+'+f.add+'</span> <span class="d">-'+f.del+'</span></span>').join("");
    let body;
    if(it.suggestion){
      body=
        '<div class="old">was '+oldMsg+'</div>'+
        '<div class="newrow"><span class="arrow">&#8594;</span>'+
        '<input class="msg" data-i="'+i+'" value="'+esc(it.suggestion.message)+'"></div>'+
        (it.suggestion.reason?'<div class="reason">'+esc(it.suggestion.reason)+'</div>':'');
    } else {
      body='<div class="kept">kept '+oldMsg+'</div>';
    }
    const toggle=it.suggestion?
      '<label class="switch">approve <input type="checkbox" data-t="'+i+'" '+(it.on?"checked":"")+'></label>':'';
    card.innerHTML=
      '<div class="crow"><span class="hash">'+esc(it.short)+'</span>'+badge+merge+
      '<span class="meta">'+esc(it.author)+' &middot; '+esc((it.date||"").slice(0,10))+'</span>'+toggle+'</div>'+
      body+
      (files?'<div class="files">'+files+'</div>':'')+
      (it.diff?'<button class="difftoggle" data-d="'+i+'">view diff</button><pre class="diff" id="diff-'+i+'">'+colorDiff(it.diff)+'</pre>':'');
    list.appendChild(card);
  });
  bind();
  counts();
}
function bind(){
  document.querySelectorAll("input.msg").forEach(el=>el.oninput=e=>{items[e.target.dataset.i].suggestion.message=e.target.value;counts();});
  document.querySelectorAll("input[data-t]").forEach(el=>el.onchange=e=>{items[e.target.dataset.t].on=e.target.checked;render();});
  document.querySelectorAll("button[data-d]").forEach(el=>el.onclick=e=>{const p=document.getElementById("diff-"+e.target.dataset.d);const open=p.style.display==="block";p.style.display=open?"none":"block";e.target.textContent=open?"view diff":"hide diff";});
}
function counts(){
  const sugg=items.filter(i=>i.suggestion).length;
  const appr=items.filter(i=>i.suggestion&&i.on&&i.suggestion.message.trim()).length;
  document.getElementById("c-total").textContent=items.length;
  document.getElementById("c-sugg").textContent=sugg;
  document.getElementById("c-appr").textContent=appr;
  document.getElementById("applycount").textContent=appr+(appr===1?" commit will be rewritten":" commits will be rewritten");
  document.getElementById("apply").disabled=appr===0;
}
document.getElementById("all").onclick=()=>{items.forEach(i=>{if(i.suggestion)i.on=true;});render();};
document.getElementById("none").onclick=()=>{items.forEach(i=>{if(i.suggestion)i.on=false;});render();};
document.getElementById("apply").onclick=async()=>{
  const approved=items.filter(i=>i.suggestion&&i.on&&i.suggestion.message.trim()).map(i=>({hash:i.hash,message:i.suggestion.message.trim()}));
  const btn=document.getElementById("apply");btn.disabled=true;btn.textContent="Rewriting...";
  const r=await fetch("/apply",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({approved})});
  const d=await r.json();
  const m=document.getElementById("modal");
  if(d.ok){
    m.innerHTML='<h2><span class="ok-ico">&#10003;</span> history rewritten</h2>'+
      '<div class="row">Rewrote <b>'+d.result.rewritten+'</b> commit messages.</div>'+
      '<div class="row">New HEAD: <code>'+esc(d.result.new_head)+'</code></div>'+
      '<div class="row">Backup branch: <code>'+esc(d.result.backup)+'</code></div>'+
      '<div class="row">Undo anytime with <code>git reset --hard '+esc(d.result.backup)+'</code></div>'+
      '<div class="row"><button class="btn-primary" onclick="location.reload()">Reload</button></div>';
  } else {
    m.innerHTML='<h2><span class="err-ico">&#10007;</span> rewrite failed</h2>'+
      '<div class="row"><code>'+esc(d.error)+'</code></div>'+
      '<div class="row"><button onclick="document.getElementById(\'overlay\').style.display=\'none\'">Close</button></div>';
    btn.disabled=false;btn.textContent="Apply approved";
  }
  document.getElementById("overlay").style.display="flex";
};
fetch("/data").then(r=>r.json()).then(d=>{
  document.getElementById("repo").textContent=d.repo;
  items=d.items.map(i=>Object.assign(i,{on:!!i.suggestion}));
  render();
});
</script>
</body>
</html>"""


USAGE = """usage:
  fix_git_history.py collect <repo> <out.json> [count]
  fix_git_history.py serve <repo> <commits.json> <suggestions.json> [port]"""


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "_msgfilter":
        msgfilter()
        return
    if cmd == "collect":
        if len(sys.argv) < 4:
            print(USAGE)
            sys.exit(1)
        count = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        collect(sys.argv[2], sys.argv[3], count)
        return
    if cmd == "serve":
        if len(sys.argv) < 5:
            print(USAGE)
            sys.exit(1)
        port = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        serve(sys.argv[2], sys.argv[3], sys.argv[4], port)
        return
    print(USAGE)
    sys.exit(1)


if __name__ == "__main__":
    main()
