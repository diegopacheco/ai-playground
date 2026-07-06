import argparse
import datetime
import html
import json
import re
from pathlib import Path


def checked_lines(value, field):
    if not isinstance(value, list):
        raise SystemExit(f"{field} must contain 3 to 7 lines")
    lines = [str(line).strip() for line in value if str(line).strip()]
    if not 3 <= len(lines) <= 7:
        raise SystemExit(f"{field} must contain 3 to 7 lines")
    return lines


def render_items(items):
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    project = str(data.get("project") or "project")
    agent = str(data.get("agent") or "Unknown")
    bugs = data.get("bugs", [])
    if not isinstance(bugs, list):
        raise SystemExit("bugs must be a list")
    cards = []
    for index, item in enumerate(bugs, 1):
        bug = checked_lines(item.get("bug"), f"bugs[{index}].bug")
        solution = checked_lines(item.get("solution"), f"bugs[{index}].solution")
        search = html.escape(" ".join(bug + solution).lower(), quote=True)
        cards.append(f'<article class="card" data-search="{search}"><div class="number">{index:02d}</div><section><h2>Bug</h2><ul>{render_items(bug)}</ul></section><section class="solution"><h2>Solution</h2><ul>{render_items(solution)}</ul></section></article>')
    safe_project = re.sub(r"[^A-Za-z0-9._-]+", "-", project).strip("-") or "project"
    date = datetime.datetime.now().astimezone().strftime("%d_%m_%Y")
    filename = f"{safe_project}-bug-report-{date}.html"
    content = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(project)} Bug Report</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--paper:#f7f9fc;--white:#fff;--bug:#c2413b;--fix:#147d64;--accent:#3157d5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{background:var(--white);border-bottom:1px solid var(--line)}}.wrap{{width:min(1120px,calc(100% - 40px));margin:auto}}.hero{{padding:64px 0 36px}}.eyebrow{{color:var(--accent);font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}}h1{{margin:8px 0 12px;font-size:clamp(2.2rem,6vw,4.7rem);line-height:1;letter-spacing:-.055em}}.meta{{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted)}}.pill{{border:1px solid var(--line);border-radius:999px;padding:5px 11px;background:#fbfcfe}}.tools{{display:flex;align-items:center;gap:16px;padding:24px 0}}input{{width:100%;border:1px solid #c8d2e0;border-radius:12px;padding:14px 16px;background:var(--white);color:var(--ink);font:inherit;outline:none}}input:focus{{border-color:var(--accent);box-shadow:0 0 0 3px #3157d51c}}#count{{white-space:nowrap;color:var(--muted);font-size:.9rem}}main{{padding-bottom:64px}}.grid{{display:grid;gap:18px}}.card{{position:relative;display:grid;grid-template-columns:54px 1fr 1fr;gap:28px;background:var(--white);border:1px solid var(--line);border-radius:18px;padding:26px;box-shadow:0 10px 35px #243a5d0a}}.number{{color:#9aa8ba;font-size:.8rem;font-weight:800;letter-spacing:.08em}}section{{min-width:0}}section+section{{border-left:1px solid var(--line);padding-left:28px}}h2{{margin:0 0 12px;color:var(--bug);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase}}.solution h2{{color:var(--fix)}}ul{{margin:0;padding-left:19px}}li+li{{margin-top:7px}}.empty{{padding:56px;text-align:center;color:var(--muted);background:var(--white);border:1px solid var(--line);border-radius:18px}}footer{{padding:0 0 36px;color:var(--muted);font-size:.82rem}}@media(max-width:760px){{.card{{grid-template-columns:1fr;gap:16px}}section+section{{border-left:0;border-top:1px solid var(--line);padding:20px 0 0}}.number{{position:absolute;right:22px;top:22px}}.tools{{align-items:stretch;flex-direction:column}}}}
</style>
</head>
<body>
<header><div class="wrap hero"><div class="eyebrow">Resolved engineering history</div><h1>{html.escape(project)}</h1><div class="meta"><span class="pill">Agent: {html.escape(agent)}</span><span class="pill">{len(bugs)} solved bugs</span><span class="pill">{date.replace('_', '/')}</span></div></div></header>
<div class="wrap tools"><input id="search" type="search" placeholder="Search bugs and solutions" aria-label="Search bugs and solutions"><span id="count">{len(bugs)} {'result' if len(bugs) == 1 else 'results'}</span></div>
<main class="wrap"><div class="grid" id="grid">{''.join(cards) if cards else '<div class="empty">No solved bugs met the evidence threshold.</div>'}</div></main>
<footer class="wrap">Built from prior {html.escape(agent)} sessions for this project.</footer>
<script>
const search=document.querySelector('#search');const cards=[...document.querySelectorAll('.card')];const count=document.querySelector('#count');search.addEventListener('input',()=>{{const query=search.value.trim().toLowerCase();let shown=0;cards.forEach(card=>{{const visible=card.dataset.search.includes(query);card.hidden=!visible;shown+=visible}});count.textContent=`${{shown}} result${{shown===1?'':'s'}}`}});
</script>
</body>
</html>'''
    output = Path(args.output_dir).resolve() / filename
    output.write_text(content, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
