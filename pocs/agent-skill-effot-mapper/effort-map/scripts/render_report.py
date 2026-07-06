import argparse
import html
import json
import math
from datetime import date
from pathlib import Path

METRICS = [
    ("effort", "Effort", "#6366f1", "How much the user reasoned and iterated versus one-shotting the request."),
    ("bugs", "Bugs", "#ef4444", "How much time went into chasing and fixing bugs."),
    ("experience", "Experience", "#10b981", "How much the user tweaked the UX versus generate-and-done."),
    ("architecture", "Architecture", "#f59e0b", "How much the user shaped stack and structure decisions."),
    ("copy_slop", "Copy Slop", "#8b5cf6", "How much was copied from elsewhere with no changes or taste."),
]


def clamp(value):
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def stars(pct):
    filled = int(round(pct / 20))
    return "".join("★" if i < filled else "☆" for i in range(5))


def ring(pct, color):
    radius = 52
    circ = 2 * math.pi * radius
    offset = circ * (1 - pct / 100)
    return f"""<svg viewBox="0 0 128 128" class="ring" role="img" aria-label="{pct} percent">
      <circle cx="64" cy="64" r="{radius}" class="ring-bg"/>
      <circle cx="64" cy="64" r="{radius}" stroke="{color}" stroke-width="12" fill="none"
        stroke-linecap="round" stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
        transform="rotate(-90 64 64)"/>
      <text x="64" y="60" class="ring-num">{pct}</text>
      <text x="64" y="82" class="ring-pct">%</text>
    </svg>"""


def metric_card(key, label, color, blurb, data):
    pct = clamp(data.get("score", 0))
    why = html.escape(str(data.get("why", "")).strip() or blurb)
    return f"""<article class="card">
      <div class="card-ring" style="--accent:{color}">{ring(pct, color)}</div>
      <div class="card-body">
        <div class="card-head"><h3>{html.escape(label)}</h3><span class="stars" style="color:{color}">{stars(pct)}</span></div>
        <p class="score5">{int(round(pct/20))}/5</p>
        <p class="why">{why}</p>
      </div>
    </article>"""


def time_bar(minutes):
    hours = minutes / 60
    pct = min(100, minutes / 60 * 100)
    label = f"{minutes:g} min" if minutes < 60 else f"{hours:.1f} h"
    return f"""<div class="timebar"><div class="timebar-fill" style="width:{pct:.0f}%"></div></div>
      <span class="timebar-label">{label} / session</span>"""


def fmt_tokens(value):
    value = int(value or 0)
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return str(value)


def prompts_html(prompts):
    if not prompts:
        return "<p class='empty'>No prompts captured.</p>"
    items = "".join(f"<li>{html.escape(str(p).strip())}</li>" for p in prompts if str(p).strip())
    return f"<ol class='prompts'>{items}</ol>"


def build(data):
    project = html.escape(str(data.get("project", "project")))
    agent = html.escape(str(data.get("agent", "Agent")))
    generated = html.escape(str(data.get("generated") or date.today().isoformat()))
    stats = data.get("stats", {})
    metrics = data.get("metrics", {})
    prompts = data.get("prompts", [])
    sessions = data.get("sessions_analyzed", stats.get("sessions_matched", 0))
    avg_tokens = stats.get("avg_tokens", 0)
    total_tokens = stats.get("total_tokens", 0)
    avg_time = stats.get("avg_time_minutes", 0)
    cards = "".join(metric_card(k, l, c, b, metrics.get(k, {})) for k, l, c, b in METRICS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Effort Map — {project}</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f6f7fb; color:#1e2230; line-height:1.55; }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 40px 24px 72px; }}
  header {{ display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:32px; }}
  h1 {{ font-size: 2rem; margin:0; letter-spacing:-.02em; }}
  h1 span {{ color:#6366f1; }}
  .meta {{ color:#6b7280; font-size:.9rem; }}
  .badge {{ display:inline-block; background:#eef2ff; color:#4338ca; border-radius:999px;
    padding:4px 12px; font-size:.78rem; font-weight:600; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin-bottom:36px; }}
  .stat {{ background:#fff; border:1px solid #e7e9f2; border-radius:16px; padding:20px 22px;
    box-shadow:0 1px 2px rgba(20,24,50,.04); }}
  .stat .k {{ font-size:.76rem; text-transform:uppercase; letter-spacing:.06em; color:#8a90a6; font-weight:600; }}
  .stat .v {{ font-size:1.9rem; font-weight:700; margin-top:6px; letter-spacing:-.02em; }}
  .stat .sub {{ font-size:.8rem; color:#9aa0b4; margin-top:2px; }}
  .timebar {{ height:10px; background:#eef0f7; border-radius:999px; overflow:hidden; margin-top:10px; }}
  .timebar-fill {{ height:100%; background:linear-gradient(90deg,#10b981,#059669); border-radius:999px; }}
  .timebar-label {{ font-size:.8rem; color:#6b7280; }}
  h2 {{ font-size:1.15rem; margin:8px 0 18px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; }}
  .card {{ display:flex; gap:18px; align-items:center; background:#fff; border:1px solid #e7e9f2;
    border-radius:18px; padding:20px; box-shadow:0 1px 2px rgba(20,24,50,.04); }}
  .card-ring {{ flex:0 0 96px; }}
  .ring {{ width:96px; height:96px; }}
  .ring-bg {{ fill:none; stroke:#eef0f7; stroke-width:12; }}
  .ring-num {{ font-size:34px; font-weight:700; text-anchor:middle; fill:#1e2230; }}
  .ring-pct {{ font-size:14px; text-anchor:middle; fill:#9aa0b4; }}
  .card-head {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}
  .card h3 {{ margin:0; font-size:1.05rem; }}
  .stars {{ font-size:.95rem; letter-spacing:1px; }}
  .score5 {{ margin:2px 0 6px; font-size:.8rem; color:#9aa0b4; font-weight:600; }}
  .why {{ margin:0; font-size:.9rem; color:#4b5165; }}
  section {{ margin-top:40px; }}
  .prompts {{ background:#fff; border:1px solid #e7e9f2; border-radius:16px; padding:16px 16px 16px 40px;
    margin:0; max-height:420px; overflow:auto; }}
  .prompts li {{ padding:8px 4px; border-bottom:1px dashed #edeff3; font-size:.9rem; }}
  .prompts li:last-child {{ border-bottom:none; }}
  .empty {{ color:#9aa0b4; }}
  footer {{ margin-top:48px; text-align:center; color:#9aa0b4; font-size:.8rem; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Effort <span>Map</span></h1>
      <p class="meta">{project} · scored from {agent} sessions · {generated}</p>
    </div>
    <span class="badge">{sessions} sessions analyzed</span>
  </header>

  <div class="stats">
    <div class="stat"><div class="k">Prompts</div><div class="v">{len(prompts)}</div><div class="sub">captured across sessions</div></div>
    <div class="stat"><div class="k">Avg tokens / session</div><div class="v">{fmt_tokens(avg_tokens)}</div><div class="sub">{fmt_tokens(total_tokens)} total</div></div>
    <div class="stat"><div class="k">Avg time / session</div><div class="v">{avg_time:g}m</div>{time_bar(avg_time)}</div>
    <div class="stat"><div class="k">Agent</div><div class="v" style="font-size:1.4rem">{agent}</div><div class="sub">{project}</div></div>
  </div>

  <h2>Effort scorecard</h2>
  <div class="grid">{cards}</div>

  <section>
    <h2>Prompts ({len(prompts)})</h2>
    {prompts_html(prompts)}
  </section>

  <footer>Generated by the effort-map skill · {generated}</footer>
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    today = date.today()
    name = f"effort-{today.strftime('%d-%m-%Y')}-report.html"
    out = Path(args.output_dir).resolve() / name
    out.write_text(build(data), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
