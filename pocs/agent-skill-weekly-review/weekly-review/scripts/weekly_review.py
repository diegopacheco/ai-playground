import argparse
import subprocess
import sys
import os
import json
import html
import math
import re
import urllib.request
from datetime import datetime, timedelta

COLORS = ['#6366f1', '#8b5cf6', '#a855f7', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#ef4444']


def run_git(repo, args):
    return subprocess.check_output(['git', '-C', repo] + args, stderr=subprocess.DEVNULL).decode('utf-8', 'replace')


def ext_of(path):
    base = path.rsplit('/', 1)[-1]
    if '.' in base:
        return base.rsplit('.', 1)[-1].lower()
    return base or 'other'


def daterange(sd, ud):
    days = []
    d = sd
    while d <= ud:
        days.append(d)
        d += timedelta(days=1)
    return days


def collect_git(repo, since, until, author):
    info = {'name': os.path.basename(os.path.abspath(repo)), 'branch': ''}
    try:
        info['branch'] = run_git(repo, ['rev-parse', '--abbrev-ref', 'HEAD']).strip()
    except Exception:
        pass
    try:
        top = run_git(repo, ['rev-parse', '--show-toplevel']).strip()
        if top:
            info['name'] = os.path.basename(top)
    except Exception:
        pass

    fmt = '@@C@@%x1f%H%x1f%h%x1f%an%x1f%aI%x1f%s'
    cmd = ['log', '--since', since + ' 00:00:00', '--until', until + ' 23:59:59',
           '--date=iso-strict', '--pretty=tformat:' + fmt, '--numstat']
    if author:
        cmd += ['--author', author]
    try:
        out = run_git(repo, cmd)
    except Exception:
        out = ''

    commits = []
    cur = None
    for line in out.split('\n'):
        if line.startswith('@@C@@'):
            p = line.split('\x1f')
            cur = {'hash': p[1], 'short': p[2], 'author': p[3], 'date': p[4],
                   'subject': p[5] if len(p) > 5 else '', 'ins': 0, 'del': 0, 'files': []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            seg = line.split('\t')
            if len(seg) == 3:
                ins = 0 if seg[0] == '-' else int(seg[0] or 0)
                dele = 0 if seg[1] == '-' else int(seg[1] or 0)
                cur['ins'] += ins
                cur['del'] += dele
                cur['files'].append(seg[2])

    sd = datetime.strptime(since, '%Y-%m-%d').date()
    ud = datetime.strptime(until, '%Y-%m-%d').date()
    days = daterange(sd, ud)
    per_day = {d.isoformat(): 0 for d in days}
    authors = {}
    file_changes = {}
    file_types = {}
    paths_seen = set()
    total_ins = 0
    total_del = 0

    for c in commits:
        dkey = c['date'][:10]
        if dkey in per_day:
            per_day[dkey] += 1
        authors[c['author']] = authors.get(c['author'], 0) + 1
        total_ins += c['ins']
        total_del += c['del']
        for f in c['files']:
            paths_seen.add(f)
            file_changes[f] = file_changes.get(f, 0) + 1
            e = ext_of(f)
            file_types[e] = file_types.get(e, 0) + 1

    per_day_list = [{'label': d.strftime('%a'), 'sub': d.strftime('%m/%d'),
                     'date': d.isoformat(), 'count': per_day[d.isoformat()]} for d in days]
    top_files = sorted(file_changes.items(), key=lambda kv: kv[1], reverse=True)[:8]
    top_types = sorted(file_types.items(), key=lambda kv: kv[1], reverse=True)[:8]
    top_authors = sorted(authors.items(), key=lambda kv: kv[1], reverse=True)

    return {
        'info': info,
        'total_commits': len(commits),
        'insertions': total_ins,
        'deletions': total_del,
        'files_changed': len(paths_seen),
        'active_days': sum(1 for v in per_day.values() if v > 0),
        'per_day': per_day_list,
        'top_files': [{'path': p, 'count': n} for p, n in top_files],
        'top_types': [{'ext': e, 'count': n} for e, n in top_types],
        'authors': [{'name': a, 'count': n} for a, n in top_authors],
        'commits': commits,
    }


def parse_dt(val):
    val = val.strip()
    m = re.match(r'(\d{8})T(\d{6})', val)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S'), True
    if re.match(r'^\d{8}$', val):
        return datetime.strptime(val, '%Y%m%d'), False
    return None, False


def unescape_text(v):
    return v.replace('\\,', ',').replace('\\;', ';').replace('\\n', ' ').replace('\\N', ' ').strip()


def parse_ics(text):
    lines = []
    for raw in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        if raw[:1] in (' ', '\t') and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    events = []
    cur = None
    for line in lines:
        if line == 'BEGIN:VEVENT':
            cur = {}
        elif line == 'END:VEVENT':
            if cur is not None:
                events.append(cur)
            cur = None
        elif cur is not None and ':' in line:
            key, val = line.split(':', 1)
            name = key.split(';', 1)[0].upper()
            if name in ('SUMMARY', 'DTSTART', 'DTEND', 'LOCATION'):
                cur[name] = val
    return events


def collect_calendar(text, sd, ud):
    days = daterange(sd, ud)
    per_day = {d.isoformat(): 0 for d in days}
    events = []
    total_minutes = 0
    for e in parse_ics(text):
        if 'DTSTART' not in e:
            continue
        start, timed = parse_dt(e['DTSTART'])
        if start is None or not (sd <= start.date() <= ud):
            continue
        minutes = None
        if timed and 'DTEND' in e:
            end, _ = parse_dt(e['DTEND'])
            if end is not None:
                minutes = max(0, int((end - start).total_seconds() // 60))
        if minutes:
            total_minutes += minutes
            per_day[start.date().isoformat()] += minutes
        events.append({
            'summary': unescape_text(e.get('SUMMARY', '(no title)')),
            'start': start.isoformat(),
            'date': start.date().isoformat(),
            'when': start.strftime('%a %H:%M') if timed else start.strftime('%a') + ' all-day',
            'timed': timed,
            'minutes': minutes,
            'location': unescape_text(e.get('LOCATION', '')),
        })
    events.sort(key=lambda x: x['start'])
    per_day_list = [{'label': d.strftime('%a'), 'sub': d.strftime('%m/%d'),
                     'date': d.isoformat(), 'minutes': per_day[d.isoformat()]} for d in days]
    top_events = sorted([e for e in events if e['minutes']], key=lambda e: e['minutes'], reverse=True)[:6]
    busiest = max(per_day_list, key=lambda x: x['minutes']) if per_day_list else None
    return {
        'total_events': len(events),
        'total_minutes': total_minutes,
        'per_day': per_day_list,
        'events': events,
        'top_events': top_events,
        'busiest_day': busiest['label'] if busiest and busiest['minutes'] else None,
    }


def build_timeline(git, cal):
    items = []
    for c in git['commits']:
        items.append({'type': 'commit', 'when': c['date'],
                      'title': c['subject'], 'short': c['short'],
                      'meta': '+%d / -%d' % (c['ins'], c['del']), 'author': c['author']})
    for e in cal['events']:
        items.append({'type': 'event', 'when': e['start'],
                      'title': e['summary'], 'short': '',
                      'meta': ('%d min' % e['minutes']) if e['minutes'] else 'all-day',
                      'author': e['location']})
    items.sort(key=lambda x: x['when'], reverse=True)
    return items[:50]


def auto_highlights(git, cal):
    parts = ['%d commits across %d active day%s' % (
        git['total_commits'], git['active_days'], '' if git['active_days'] == 1 else 's')]
    parts.append('+%s / -%s lines over %s files' % (
        f"{git['insertions']:,}", f"{git['deletions']:,}", f"{git['files_changed']:,}"))
    if cal['total_events']:
        parts.append('%d meetings totalling %.1f hours' % (cal['total_events'], cal['total_minutes'] / 60))
    return ', '.join(parts) + '.'


def esc(s):
    return html.escape(str(s))


def bar_chart(items, fmt, accent):
    items = list(items)
    if not items:
        return ''
    w, h, pad = 600, 240, 34
    n = len(items)
    maxv = max((v for _, v in items), default=0) or 1
    gap = 16
    bw = (w - 2 * pad - gap * (n - 1)) / n
    parts = ['<svg class="chart" viewBox="0 0 %d %d" preserveAspectRatio="xMidYMid meet">' % (w, h)]
    for gy in range(1, 4):
        y = pad + (h - 2 * pad) * gy / 4
        parts.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="#eef1f8"/>' % (pad, y, w - pad, y))
    x = pad
    for i, (label, v) in enumerate(items):
        bh = (h - 2 * pad) * (v / maxv)
        y = h - pad - bh
        parts.append('<rect class="bar" style="animation-delay:%dms;fill:%s" x="%g" y="%g" width="%g" height="%g" rx="7"/>'
                     % (i * 80, accent, x, y, bw, max(bh, 3)))
        parts.append('<text class="bar-v" x="%g" y="%g">%s</text>' % (x + bw / 2, y - 8, esc(fmt(v))))
        parts.append('<text class="bar-l" x="%g" y="%g">%s</text>' % (x + bw / 2, h - pad + 18, esc(label)))
        x += bw + gap
    parts.append('</svg>')
    return ''.join(parts)


def donut(items):
    items = [(l, v) for l, v in items if v > 0]
    if not items:
        return ''
    total = sum(v for _, v in items)
    size, sw = 200, 30
    r = (size - sw) / 2 - 4
    c = 2 * math.pi * r
    cx = cy = size / 2
    parts = ['<svg class="donut" viewBox="0 0 %d %d">' % (size, size)]
    parts.append('<circle cx="%g" cy="%g" r="%g" fill="none" stroke="#eef1f8" stroke-width="%g"/>' % (cx, cy, r, sw))
    start = 0.0
    for i, (l, v) in enumerate(items):
        seg = c * (v / total)
        parts.append('<circle class="seg" cx="%g" cy="%g" r="%g" fill="none" stroke="%s" stroke-width="%g" stroke-dasharray="%.3f %.3f" stroke-dashoffset="%.3f" transform="rotate(-90 %g %g)"/>'
                     % (cx, cy, r, COLORS[i % len(COLORS)], sw, seg, c - seg, -start, cx, cy))
        start += seg
    parts.append('<text class="donut-c" x="%g" y="%g">%d</text>' % (cx, cy - 2, total))
    parts.append('<text class="donut-s" x="%g" y="%g">files</text>' % (cx, cy + 17))
    parts.append('</svg>')
    return ''.join(parts)


def legend(items):
    rows = []
    for i, (l, v) in enumerate([(l, v) for l, v in items if v > 0]):
        rows.append('<div class="lg"><span class="dot" style="background:%s"></span><span class="lg-l">%s</span><span class="lg-v">%d</span></div>'
                    % (COLORS[i % len(COLORS)], esc(l), v))
    return ''.join(rows)


def heatmap(per_day, key):
    maxv = max((d[key] for d in per_day), default=0) or 1
    cells = []
    for d in per_day:
        v = d[key]
        op = 0.12 + 0.88 * (v / maxv) if v else 0.0
        bg = ('rgba(99,102,241,%.2f)' % op) if v else '#eef1f8'
        cells.append('<div class="hc"><div class="hcell" style="background:%s"></div><div class="hl">%s</div></div>'
                     % (bg, esc(d['label'])))
    return '<div class="heat">' + ''.join(cells) + '</div>'


def stat_card(label, to, sub, accent, dec=False):
    d = ' data-dec="1"' if dec else ''
    return ('<div class="stat reveal"><div class="stat-bar" style="background:%s"></div>'
            '<div class="stat-num num" data-to="%s"%s>0</div>'
            '<div class="stat-lbl">%s</div><div class="stat-sub">%s</div></div>'
            % (accent, to, d, esc(label), esc(sub)))


CSS = """
:root{--ink:#0f172a;--mut:#64748b;--card:rgba(255,255,255,.72);--line:#e9edf6;
--g1:#6366f1;--g2:#a855f7;--g3:#ec4899;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--ink);
background:linear-gradient(135deg,#f6f8ff 0%,#fdf4fb 55%,#f3fbff 100%);min-height:100vh;
-webkit-font-smoothing:antialiased;overflow-x:hidden}
.bg-blob{position:fixed;border-radius:50%;filter:blur(80px);opacity:.45;z-index:0;pointer-events:none}
.b1{width:520px;height:520px;background:#c7d2fe;top:-160px;right:-120px}
.b2{width:460px;height:460px;background:#fbcfe8;bottom:-180px;left:-140px}
.b3{width:380px;height:380px;background:#bae6fd;top:40%;left:50%}
.wrap{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:56px 24px 80px}
.hero{text-align:center;margin-bottom:52px}
.kicker{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:600;letter-spacing:.14em;
text-transform:uppercase;color:#7c3aed;background:#ede9fe;padding:7px 16px;border-radius:999px;margin-bottom:22px}
.kicker .pulse{width:8px;height:8px;border-radius:50%;background:#7c3aed;box-shadow:0 0 0 0 rgba(124,58,237,.5);animation:pulse 2s infinite}
h1{font-family:'Space Grotesk','Inter',sans-serif;font-size:clamp(40px,6vw,68px);font-weight:700;line-height:1.04;letter-spacing:-.02em}
.grad{background:linear-gradient(110deg,var(--g1),var(--g2) 45%,var(--g3));-webkit-background-clip:text;background-clip:text;color:transparent}
.range{margin-top:14px;font-size:18px;color:var(--mut);font-weight:500}
.repo{margin-top:6px;font-size:15px;color:var(--mut)}
.repo b{color:var(--ink)}
.hl{max-width:760px;margin:26px auto 0;font-size:18px;line-height:1.65;color:#334155;
background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px 26px;
backdrop-filter:blur(10px);box-shadow:0 12px 40px -22px rgba(99,102,241,.5)}
.grid{display:grid;gap:18px}
.stats{grid-template-columns:repeat(auto-fit,minmax(165px,1fr));margin-bottom:40px}
.stat{position:relative;background:var(--card);border:1px solid var(--line);border-radius:20px;padding:26px 22px;
overflow:hidden;backdrop-filter:blur(10px);box-shadow:0 14px 40px -26px rgba(30,41,59,.45)}
.stat-bar{position:absolute;left:0;top:0;bottom:0;width:5px}
.stat-num{font-family:'Space Grotesk','Inter',sans-serif;font-size:42px;font-weight:700;line-height:1;letter-spacing:-.02em}
.stat-lbl{margin-top:10px;font-size:14px;font-weight:600;color:var(--ink)}
.stat-sub{margin-top:3px;font-size:12.5px;color:var(--mut)}
.section{margin-top:14px;margin-bottom:18px;display:flex;align-items:baseline;gap:12px}
.section h2{font-family:'Space Grotesk','Inter',sans-serif;font-size:24px;font-weight:600;letter-spacing:-.01em}
.section .line{flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.cols{display:grid;grid-template-columns:1.45fr 1fr;gap:18px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:26px;
backdrop-filter:blur(10px);box-shadow:0 16px 46px -30px rgba(30,41,59,.5)}
.card h3{font-size:15px;font-weight:600;color:var(--mut);margin-bottom:18px;letter-spacing:.02em}
.chart{width:100%;height:auto}
.bar{transform:scaleY(0);transform-origin:bottom;transform-box:fill-box;animation:grow .8s cubic-bezier(.22,1,.36,1) forwards}
.bar-v{fill:var(--ink);font-size:14px;font-weight:700;text-anchor:middle;font-family:'Space Grotesk',sans-serif}
.bar-l{fill:var(--mut);font-size:13px;text-anchor:middle}
.donut-wrap{display:flex;align-items:center;gap:22px;flex-wrap:wrap;justify-content:center}
.donut{width:200px;height:200px;flex:none}
.donut-c{fill:var(--ink);font-size:38px;font-weight:700;text-anchor:middle;font-family:'Space Grotesk',sans-serif}
.donut-s{fill:var(--mut);font-size:13px;text-anchor:middle}
.legend{display:flex;flex-direction:column;gap:9px;min-width:150px}
.lg{display:flex;align-items:center;gap:10px;font-size:14px}
.dot{width:11px;height:11px;border-radius:3px;flex:none}
.lg-l{flex:1;color:#334155}.lg-v{font-weight:700;color:var(--ink)}
.heat{display:grid;grid-template-columns:repeat(7,1fr);gap:10px}
.hc{text-align:center}
.hcell{height:54px;border-radius:12px;border:1px solid var(--line)}
.hl{margin-top:7px;font-size:12px;color:var(--mut)}
.list{display:flex;flex-direction:column}
.row{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px dashed var(--line)}
.row:last-child{border-bottom:none}
.rk{width:26px;height:26px;border-radius:8px;background:#eef2ff;color:#6366f1;font-weight:700;font-size:13px;
display:flex;align-items:center;justify-content:center;flex:none}
.rt{flex:1;font-size:14px;color:#1e293b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,Menlo,monospace}
.rv{font-weight:700;font-size:13px;color:var(--mut)}
.tl{position:relative;padding-left:30px}
.tl:before{content:'';position:absolute;left:9px;top:6px;bottom:6px;width:2px;
background:linear-gradient(var(--g1),var(--g3))}
.ti{position:relative;padding:13px 0}
.ti:before{content:'';position:absolute;left:-25px;top:18px;width:14px;height:14px;border-radius:50%;
border:3px solid #fff;box-shadow:0 0 0 2px var(--g1)}
.ti.ev:before{box-shadow:0 0 0 2px var(--g3);background:var(--g3)}
.ti.cm:before{background:var(--g1)}
.ti-h{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tag{font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:3px 9px;border-radius:999px}
.tag.cm{background:#eef2ff;color:#4f46e5}.tag.ev{background:#fce7f3;color:#db2777}
.ti-w{font-size:12.5px;color:var(--mut)}
.ti-t{margin-top:5px;font-size:15px;color:#1e293b;font-weight:500}
.ti-m{margin-top:3px;font-size:12.5px;color:var(--mut);font-family:ui-monospace,Menlo,monospace}
.foot{text-align:center;margin-top:60px;font-size:13px;color:#94a3b8}
.reveal{opacity:0;transform:translateY(18px);transition:opacity .6s ease,transform .6s cubic-bezier(.22,1,.36,1)}
.reveal.in{opacity:1;transform:none}
@keyframes grow{to{transform:scaleY(1)}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(124,58,237,.5)}70%{box-shadow:0 0 0 10px rgba(124,58,237,0)}100%{box-shadow:0 0 0 0 rgba(124,58,237,0)}}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
"""

JS = """
(function(){
var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}});},{threshold:.12});
document.querySelectorAll('.reveal').forEach(function(el){io.observe(el);});
function count(el){var to=parseFloat(el.getAttribute('data-to'))||0;var dec=el.getAttribute('data-dec')==='1';var t0=null;
function step(ts){if(!t0)t0=ts;var p=Math.min(1,(ts-t0)/900);var e=1-Math.pow(1-p,3);var v=to*e;
el.textContent=dec?v.toFixed(1):Math.round(v).toLocaleString();if(p<1)requestAnimationFrame(step);}requestAnimationFrame(step);}
var io2=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){count(e.target);io2.unobserve(e.target);}});},{threshold:.5});
document.querySelectorAll('.num').forEach(function(el){io2.observe(el);});
})();
"""


def render(git, cal, label, highlights):
    has_cal = cal['total_events'] > 0
    p = []
    p.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    p.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    p.append('<title>Weekly Review &middot; %s</title>' % esc(git['info']['name']))
    p.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    p.append('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">')
    p.append('<style>%s</style></head><body>' % CSS)
    p.append('<div class="bg-blob b1"></div><div class="bg-blob b2"></div><div class="bg-blob b3"></div>')
    p.append('<div class="wrap">')

    p.append('<div class="hero">')
    p.append('<div class="kicker"><span class="pulse"></span>Weekly Review</div>')
    p.append('<h1>Your week in<br><span class="grad">code &amp; calendar</span></h1>')
    p.append('<div class="range">%s</div>' % esc(label))
    p.append('<div class="repo"><b>%s</b> &middot; branch %s</div>' % (esc(git['info']['name']), esc(git['info']['branch'] or 'n/a')))
    p.append('<div class="hl reveal">%s</div>' % esc(highlights))
    p.append('</div>')

    p.append('<div class="grid stats">')
    p.append(stat_card('Commits', git['total_commits'], 'this week', 'linear-gradient(var(--g1),var(--g2))'))
    p.append(stat_card('Lines added', git['insertions'], 'insertions', 'linear-gradient(#10b981,#06b6d4)'))
    p.append(stat_card('Lines removed', git['deletions'], 'deletions', 'linear-gradient(#f59e0b,#ef4444)'))
    p.append(stat_card('Files touched', git['files_changed'], 'unique paths', 'linear-gradient(var(--g2),var(--g3))'))
    if has_cal:
        p.append(stat_card('Meetings', cal['total_events'], 'calendar events', 'linear-gradient(#ec4899,#8b5cf6)'))
        p.append(stat_card('Meeting hours', round(cal['total_minutes'] / 60, 1), 'in calls', 'linear-gradient(#06b6d4,#6366f1)', dec=True))
    else:
        p.append(stat_card('Active days', git['active_days'], 'with commits', 'linear-gradient(#06b6d4,#6366f1)'))
    p.append('</div>')

    p.append('<div class="section reveal"><h2>Commit rhythm</h2><div class="line"></div></div>')
    p.append('<div class="cols">')
    p.append('<div class="card reveal"><h3>COMMITS PER DAY</h3>%s</div>'
             % bar_chart([(d['label'], d['count']) for d in git['per_day']], lambda v: str(int(v)), 'url(#g)' if False else '#6366f1'))
    p.append('<div class="card reveal"><h3>FILE TYPES</h3><div class="donut-wrap">%s<div class="legend">%s</div></div></div>'
             % (donut([(t['ext'], t['count']) for t in git['top_types']]),
                legend([(t['ext'], t['count']) for t in git['top_types']])))
    p.append('</div>')

    p.append('<div class="card reveal" style="margin-bottom:14px"><h3>ACTIVITY HEATMAP</h3>%s</div>'
             % heatmap(git['per_day'], 'count'))

    if git['top_files'] or git['authors']:
        p.append('<div class="cols">')
        rows = ''.join('<div class="row"><div class="rk">%d</div><div class="rt">%s</div><div class="rv">%d&times;</div></div>'
                       % (i + 1, esc(f['path']), f['count']) for i, f in enumerate(git['top_files']))
        p.append('<div class="card reveal"><h3>MOST CHANGED FILES</h3><div class="list">%s</div></div>'
                 % (rows or '<div class="rt">No changes</div>'))
        arows = ''.join('<div class="row"><div class="rk">%d</div><div class="rt" style="font-family:inherit">%s</div><div class="rv">%d commits</div></div>'
                        % (i + 1, esc(a['name']), a['count']) for i, a in enumerate(git['authors'][:8]))
        p.append('<div class="card reveal"><h3>CONTRIBUTORS</h3><div class="list">%s</div></div>' % arows)
        p.append('</div>')

    if has_cal:
        p.append('<div class="section reveal"><h2>Where the time went</h2><div class="line"></div></div>')
        p.append('<div class="cols">')
        p.append('<div class="card reveal"><h3>MEETING HOURS PER DAY</h3>%s</div>'
                 % bar_chart([(d['label'], d['minutes']) for d in cal['per_day']], lambda v: ('%.1f' % (v / 60)) if v else '0', '#ec4899'))
        trows = ''.join('<div class="row"><div class="rk">%d</div><div class="rt" style="font-family:inherit">%s</div><div class="rv">%.1fh</div></div>'
                        % (i + 1, esc(e['summary']), e['minutes'] / 60) for i, e in enumerate(cal['top_events']))
        p.append('<div class="card reveal"><h3>LONGEST MEETINGS</h3><div class="list">%s</div></div>'
                 % (trows or '<div class="rt">No timed meetings</div>'))
        p.append('</div>')

    timeline = build_timeline(git, cal)
    if timeline:
        p.append('<div class="section reveal"><h2>The week, end to end</h2><div class="line"></div></div>')
        p.append('<div class="card reveal"><div class="tl">')
        for it in timeline:
            cls = 'ev' if it['type'] == 'event' else 'cm'
            tag = 'Meeting' if it['type'] == 'event' else 'Commit'
            when = it['when'][:16].replace('T', ' ')
            extra = (' &middot; ' + esc(it['short'])) if it['short'] else ''
            sub = (' &middot; ' + esc(it['author'])) if it['author'] else ''
            p.append('<div class="ti %s"><div class="ti-h"><span class="tag %s">%s</span>'
                     '<span class="ti-w">%s%s</span></div><div class="ti-t">%s</div>'
                     '<div class="ti-m">%s%s</div></div>'
                     % (cls, cls, tag, esc(when), sub, esc(it['title'] or '(no title)'), esc(it['meta']), extra))
        p.append('</div></div>')

    p.append('<div class="foot">Generated by the weekly-review skill &middot; git history + Google Calendar</div>')
    p.append('</div><script>%s</script></body></html>' % JS)
    return ''.join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--since')
    ap.add_argument('--until')
    ap.add_argument('--ical-url')
    ap.add_argument('--ical-file')
    ap.add_argument('--author')
    ap.add_argument('--highlights')
    ap.add_argument('--out', default='./weekly-review-site')
    a = ap.parse_args()

    today = datetime.now().date()
    until = a.until or today.isoformat()
    since = a.since or (datetime.strptime(until, '%Y-%m-%d').date() - timedelta(days=6)).isoformat()
    sd = datetime.strptime(since, '%Y-%m-%d').date()
    ud = datetime.strptime(until, '%Y-%m-%d').date()
    label = '%s – %s' % (sd.strftime('%b %d'), ud.strftime('%b %d, %Y'))

    git = collect_git(a.repo, since, until, a.author)

    ics_text = ''
    if a.ical_file:
        try:
            with open(a.ical_file, 'r', encoding='utf-8', errors='replace') as f:
                ics_text = f.read()
        except Exception as e:
            sys.stderr.write('calendar file read failed: %s\n' % e)
    elif a.ical_url:
        try:
            req = urllib.request.Request(a.ical_url, headers={'User-Agent': 'weekly-review/1.0'})
            with urllib.request.urlopen(req, timeout=20) as r:
                ics_text = r.read().decode('utf-8', 'replace')
        except Exception as e:
            sys.stderr.write('calendar fetch failed: %s\n' % e)

    cal = collect_calendar(ics_text, sd, ud) if ics_text else collect_calendar('', sd, ud)
    highlights = a.highlights or auto_highlights(git, cal)

    html_out = render(git, cal, label, highlights)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_out)

    summary = {
        'site': os.path.abspath(path),
        'range': {'since': since, 'until': until, 'label': label},
        'repo': git['info']['name'],
        'commits': git['total_commits'],
        'insertions': git['insertions'],
        'deletions': git['deletions'],
        'files_changed': git['files_changed'],
        'active_days': git['active_days'],
        'busiest_commit_day': max(git['per_day'], key=lambda d: d['count'])['label'] if git['per_day'] else None,
        'top_subjects': [c['subject'] for c in git['commits'][:6]],
        'meetings': cal['total_events'],
        'meeting_hours': round(cal['total_minutes'] / 60, 1),
        'busiest_meeting_day': cal['busiest_day'],
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
