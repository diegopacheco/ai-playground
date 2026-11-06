use factoryfile::{Factory, Stage};

use crate::grid::draw;
use crate::layout::{Layout, count, grid, layout, route_text, work};

pub(crate) const W: usize = 230;
pub(crate) const H: usize = 84;
const COL_GAP: usize = 40;
const ARC: usize = 56;
const ROW_GAP: usize = 64;
const PAD: usize = 28;
const LANE: usize = 34;
const BULGE: usize = 56;

pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    format!("{}…", s.chars().take(max - 1).collect::<String>())
}

pub(crate) struct Placed {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) layer: usize,
}

struct Geometry {
    places: Vec<Placed>,
    left: usize,
    right: usize,
    height: usize,
}

fn geometry(factory: &Factory, graph: &Layout) -> Geometry {
    let left = PAD + if graph.backward.is_empty() { 0 } else { BULGE + LANE * graph.backward.len() };
    let cols = graph.layers.iter().map(Vec::len).max().unwrap_or(0);
    let mut places: Vec<Placed> = factory.stages.iter().map(|_| Placed { x: 0, y: 0, layer: 0 }).collect();
    for (layer, stages) in graph.layers.iter().enumerate() {
        let offset = (cols - stages.len()) * (W + COL_GAP) / 2;
        for (col, stage) in stages.iter().enumerate() {
            places[*stage] = Placed {
                x: left + offset + col * (W + COL_GAP),
                y: PAD + layer * (H + ROW_GAP),
                layer,
            };
        }
    }
    let right = left + cols * W + cols.saturating_sub(1) * COL_GAP;
    let height = PAD * 2 + graph.layers.len() * H + graph.layers.len().saturating_sub(1) * ROW_GAP;
    Geometry { places, left, right, height }
}

fn lanes_by_span(spans: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by_key(|i| (spans[*i], *i));
    let mut lane = vec![0; spans.len()];
    for (rank, i) in order.into_iter().enumerate() {
        lane[i] = rank;
    }
    lane
}

pub fn svg(factory: &Factory) -> String {
    let graph = layout(factory);
    let (g, edges) = match grid(factory) {
        Some(cells) => {
            let drawn = draw(factory, &graph, &cells);
            (Geometry { places: drawn.places, left: PAD, right: drawn.right, height: drawn.height }, drawn.edges)
        }
        None => {
            let g = geometry(factory, &graph);
            let edges = layered_edges(factory, &graph, &g);
            (g, edges)
        }
    };
    let loops: Vec<(usize, usize, usize)> = factory
        .loops
        .iter()
        .enumerate()
        .filter_map(|(i, l)| Some((i, index(factory, &l.from)?, index(factory, &l.to)?)))
        .collect();
    let width = g.right + PAD + if loops.is_empty() { 0 } else { BULGE + LANE * loops.len() };
    let mut out = format!(
        "<svg class=\"graph\" viewBox=\"0 0 {width} {}\" width=\"{width}\" height=\"{}\" role=\"img\" aria-label=\"{} factory graph\">",
        g.height,
        g.height,
        escape(&factory.name)
    );
    out.push_str("<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M0 0L10 5L0 10z\" fill=\"context-stroke\"/></marker></defs>");
    out.push_str(&edges);
    let loop_spans: Vec<usize> = loops.iter().map(|(_, f, t)| g.places[*f].layer.abs_diff(g.places[*t].layer)).collect();
    for ((i, from, to), lane) in loops.iter().zip(lanes_by_span(&loop_spans)) {
        let l = &factory.loops[*i];
        let x = g.right + BULGE + LANE * lane;
        let class = if l.max.is_some() { "loop" } else { "loop open" };
        let title = format!("{} · {}", l.name, l.max.map_or("no max".to_string(), |m| format!("max {m}")));
        out.push_str(&lane_edge(&g.places[*from], &g.places[*to], x, true, class, &(i + 1).to_string(), &title, &format!("loop-{i}")));
    }
    for (i, stage) in factory.stages.iter().enumerate() {
        out.push_str(&node(factory, stage, &g.places[i], i));
    }
    out.push_str("</svg>");
    out
}

fn layered_edges(factory: &Factory, graph: &Layout, g: &Geometry) -> String {
    let mut out = String::new();
    for e in &graph.forward {
        let label = route_text(&factory.stages[e.from], e);
        let class = if label.is_some() { "edge route" } else { "edge" };
        out.push_str(&forward_edge(&g.places[e.from], &g.places[e.to], class, label.as_deref()));
    }
    for (i, stage) in factory.stages.iter().enumerate() {
        for route in &stage.routes {
            for fallback in route.fallbacks().into_iter().filter_map(|f| Some((f, index(factory, &f.to)?))) {
                let label = fallback.0.label.clone().unwrap_or_else(|| "exhausted".to_string());
                out.push_str(&forward_edge(&g.places[i], &g.places[fallback.1], "fallback", Some(&label)));
            }
        }
    }
    let back_spans: Vec<usize> = graph.backward.iter().map(|e| g.places[e.from].layer.abs_diff(g.places[e.to].layer)).collect();
    for (e, lane) in graph.backward.iter().zip(lanes_by_span(&back_spans)) {
        let from = &factory.stages[e.from];
        let via = route_text(from, e).map_or("next".to_string(), |r| format!("on {r}"));
        let (class, badge, bound) = if e.bounded { ("back bounded", "↻", "bounded") } else { ("back", "!", "unbounded") };
        let title = format!("{bound} {via} {} → {}", from.name, factory.stages[e.to].name);
        out.push_str(&lane_edge(&g.places[e.from], &g.places[e.to], g.left - BULGE - LANE * lane, false, class, badge, &title, ""));
    }
    out
}

fn index(factory: &Factory, name: &str) -> Option<usize> {
    factory.stages.iter().position(|s| s.name == name)
}

fn same_row(a: &Placed, b: &Placed) -> bool {
    a.layer == b.layer && a.x != b.x
}

fn edge_label(x: usize, y: usize, label: Option<&str>) -> String {
    label.map(|l| format!("<text class=\"edge-label\" x=\"{x}\" y=\"{y}\" text-anchor=\"middle\">{}</text>", escape(l))).unwrap_or_default()
}

fn arc(a: &Placed, b: &Placed, label: Option<&str>) -> String {
    let (x1, x2) = (a.x + W / 2, b.x + W / 2);
    let (y, ctrl, ly) = (a.y + H, a.y + H + ARC, a.y + H + ARC * 3 / 4 + 12);
    format!(
        "<path d=\"M{x1} {y} C{x1} {ctrl} {x2} {ctrl} {x2} {y}\" marker-end=\"url(#arrow)\"/>{}",
        edge_label((x1 + x2) / 2, ly, label)
    )
}

fn forward_edge(a: &Placed, b: &Placed, class: &str, label: Option<&str>) -> String {
    if same_row(a, b) && a.x.abs_diff(b.x) > W + COL_GAP {
        return format!("<g class=\"{class}\">{}</g>", arc(a, b, label));
    }
    let (d, lx, ly) = if b.layer == a.layer + 1 {
        let (x1, y1, x2, y2) = (a.x + W / 2, a.y + H, b.x + W / 2, b.y);
        let mid = (y1 + y2) / 2;
        (format!("M{x1} {y1} C{x1} {mid} {x2} {mid} {x2} {y2}"), (x1 + x2) / 2, mid)
    } else if same_row(a, b) {
        let y = a.y + H / 2;
        let (x1, x2) = if b.x > a.x { (a.x + W, b.x) } else { (a.x, b.x + W) };
        (format!("M{x1} {y} H{x2}"), (x1 + x2) / 2, y - 8)
    } else {
        let (x1, y1, x2, y2) = (a.x + W, a.y + H / 2, b.x + W, b.y + H / 2);
        let bend = a.x.max(b.x) + W + BULGE * 3 / 4;
        (format!("M{x1} {y1} C{bend} {y1} {bend} {y2} {x2} {y2}"), a.x.max(b.x) + W + BULGE / 2, (y1 + y2) / 2)
    };
    format!("<path class=\"{class}\" d=\"{d}\" marker-end=\"url(#arrow)\"/>{}", edge_label(lx, ly, label))
}

#[allow(clippy::too_many_arguments)]
fn lane_edge(a: &Placed, b: &Placed, lane: usize, right: bool, class: &str, badge: &str, title: &str, target: &str) -> String {
    let (ax, bx) = if right { (a.x + W, b.x + W) } else { (a.x, b.x) };
    let (y1, y2) = (a.y + H * 2 / 3, b.y + H / 3);
    let r = 12;
    let (out_dir, in_dir) = if right { (lane - r, lane - r) } else { (lane + r, lane + r) };
    let up = y2 < y1;
    let (c1, c2) = if up { (y1 - r, y2 + r) } else { (y1 + r, y2.saturating_sub(r)) };
    let d = format!("M{ax} {y1} H{out_dir} Q{lane} {y1} {lane} {c1} V{c2} Q{lane} {y2} {in_dir} {y2} H{bx}");
    let mid = (y1 + y2) / 2;
    let data = if target.is_empty() { String::new() } else { format!(" data-target=\"{target}\"") };
    format!(
        "<g class=\"{class}\"{data}><title>{}</title><path d=\"{d}\" marker-end=\"url(#arrow)\"/><circle cx=\"{lane}\" cy=\"{mid}\" r=\"11\"/><text x=\"{lane}\" y=\"{}\" text-anchor=\"middle\">{badge}</text></g>",
        escape(title),
        mid + 4
    )
}

fn node(factory: &Factory, stage: &Stage, at: &Placed, i: usize) -> String {
    let is_entry = factory.entry.as_deref() == Some(stage.name.as_str());
    let class = if is_entry { "node entry" } else { "node" };
    let roles: Vec<String> = stage
        .agents
        .iter()
        .map(|a| factory.agent(a).and_then(|agent| agent.role.clone()).unwrap_or_else(|| a.clone()))
        .collect();
    let roles = if stage.run.is_some() || roles.is_empty() { work(stage) } else { roles.join(", ") };
    let kind = if stage.run.is_some() { "command".to_string() } else { format!("{} agent{}", stage.agents.len(), if stage.agents.len() == 1 { "" } else { "s" }) };
    let artifacts = match stage.artifacts.len() {
        0 => String::new(),
        1 => "1 artifact".to_string(),
        n => format!("{n} artifacts"),
    };
    format!(
        "<g class=\"{class}\" data-target=\"stage-{i}\" tabindex=\"0\"><rect x=\"{}\" y=\"{}\" width=\"{W}\" height=\"{H}\" rx=\"14\"/><text class=\"name\" x=\"{}\" y=\"{}\">{}</text><text class=\"roles\" x=\"{}\" y=\"{}\">{}</text><text class=\"meta\" x=\"{}\" y=\"{}\">{}</text></g>",
        at.x,
        at.y,
        at.x + 16,
        at.y + 28,
        escape(&truncate(&format!("{}{}", if is_entry { "▶ " } else { "" }, stage.name), 24)),
        at.x + 16,
        at.y + 50,
        escape(&truncate(&roles, 30)),
        at.x + 16,
        at.y + 70,
        escape(&format!("{kind}{}", if artifacts.is_empty() { String::new() } else { format!(" · {artifacts}") })),
    )
}

fn list(items: &[String]) -> String {
    if items.is_empty() {
        return "<span class=\"none\">none</span>".to_string();
    }
    items.iter().map(|i| format!("<code>{}</code>", escape(i))).collect::<Vec<_>>().join(" ")
}

fn row(label: &str, value: &str) -> String {
    format!("<div class=\"row\"><span>{label}</span><div>{value}</div></div>")
}

fn details(factory: &Factory, slug: &str) -> String {
    let mut out = String::new();
    for (i, stage) in factory.stages.iter().enumerate() {
        let mut body = String::new();
        if let Some(d) = &stage.description {
            body.push_str(&format!("<p>{}</p>", escape(d)));
        }
        if stage.run.is_some() {
            body.push_str(&row("run", &list(&[work(stage)])));
        }
        body.push_str(&row("next", &list(&stage.next)));
        if !stage.routes.is_empty() {
            let routes: Vec<String> = stage.routes.iter().map(|r| format!("{} → {}", r.outcome, r.to)).collect();
            body.push_str(&row("routes", &list(&routes)));
            let fallbacks: Vec<String> = stage.routes.iter().flat_map(|r| r.fallbacks()).map(|f| f.to.clone()).collect();
            if !fallbacks.is_empty() {
                body.push_str(&row("fallbacks", &list(&fallbacks)));
            }
        }
        body.push_str(&row("artifacts", &list(&stage.artifacts)));
        let loops: Vec<String> = factory.loops.iter().filter(|l| l.from == stage.name || l.to == stage.name).map(|l| l.name.clone()).collect();
        body.push_str(&row("loops", &list(&loops)));
        if let Some(c) = &stage.content {
            body.push_str(&format!("<pre>{}</pre>", escape(c)));
        }
        for name in &stage.agents {
            body.push_str(&agent_card(factory, name));
        }
        out.push_str(&format!(
            "<article class=\"detail\" id=\"{slug}-stage-{i}\" hidden><h3><small>stage</small>{}</h3>{body}</article>",
            escape(&stage.name)
        ));
    }
    for (i, l) in factory.loops.iter().enumerate() {
        let mut body = String::new();
        if let Some(d) = &l.description {
            body.push_str(&format!("<p>{}</p>", escape(d)));
        }
        body.push_str(&row("from", &list(std::slice::from_ref(&l.from))));
        body.push_str(&row("back to", &list(std::slice::from_ref(&l.to))));
        body.push_str(&row("until", &escape(l.until.as_deref().unwrap_or("not set"))));
        body.push_str(&row("max", &l.max.map_or("<span class=\"warn\">no max, it may never end</span>".to_string(), |m| m.to_string())));
        out.push_str(&format!(
            "<article class=\"detail\" id=\"{slug}-loop-{i}\" hidden><h3><small>loop</small>{}</h3>{body}</article>",
            escape(&l.name)
        ));
    }
    out
}

fn agent_card(factory: &Factory, name: &str) -> String {
    let Some(agent) = factory.agent(name) else {
        return format!("<div class=\"agent\"><b>{}</b></div>", escape(name));
    };
    let mut meta = Vec::new();
    if let Some(r) = &agent.role {
        meta.push(escape(r));
    }
    if let Some(m) = &agent.model {
        meta.push(format!("<code>{}</code>", escape(m)));
    }
    let prompt = match (&agent.content, &agent.source) {
        (Some(c), _) => format!("<pre>{}</pre>", escape(c)),
        (None, Some(s)) => format!("<p class=\"source\">source <code>{}</code></p>", escape(s)),
        _ => String::new(),
    };
    format!(
        "<div class=\"agent\"><b>{}</b><span>{}</span>{}{prompt}</div>",
        escape(name),
        meta.join(" · "),
        row("tools", &list(&agent.tools))
    )
}

fn section(factory: &Factory, slug: &str, active: bool) -> String {
    let description = factory.description.as_deref().map(|d| format!("<p class=\"lede\">{}</p>", escape(d))).unwrap_or_default();
    let stats = format!(
        "<p class=\"stats\">entry <code>{}</code> · {} · {} · {}</p>",
        escape(factory.entry.as_deref().unwrap_or("none")),
        count(factory.stages.len(), "stage"),
        count(factory.agents.len(), "agent"),
        count(factory.loops.len(), "loop")
    );
    let hint = "<p class=\"hint\">Click a stage or a loop to see its agents, prompts and exit condition.</p>";
    format!(
        "<section class=\"factory\" id=\"{slug}\"{}><header><h2>{}</h2>{description}{stats}</header><div class=\"stage-area\"><div class=\"canvas\">{}{}</div><aside>{hint}{}</aside></div></section>",
        if active { "" } else { " hidden" },
        escape(&factory.name),
        svg(factory).replace("data-target=\"", &format!("data-target=\"{slug}-")),
        legend(factory, slug),
        details(factory, slug)
    )
}

fn legend(factory: &Factory, slug: &str) -> String {
    if factory.loops.is_empty() {
        return String::new();
    }
    let items: String = factory
        .loops
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let max = l.max.map_or("<b class=\"warn\">no max</b>".to_string(), |m| format!("max {m}"));
            let until = l.until.as_deref().map(|u| format!(" · until {}", escape(u))).unwrap_or_default();
            format!(
                "<li data-target=\"{slug}-loop-{i}\" tabindex=\"0\"><i>{}</i><b>{}</b> <code>{} → {}</code>{until} · {max}</li>",
                i + 1,
                escape(&l.name),
                escape(&l.from),
                escape(&l.to)
            )
        })
        .collect();
    format!("<ol class=\"legend\">{items}</ol>")
}

pub fn page(title: &str, factories: &[&Factory]) -> String {
    let slugs: Vec<String> = (0..factories.len()).map(|i| format!("f{i}")).collect();
    let tabs: String = factories
        .iter()
        .zip(&slugs)
        .enumerate()
        .map(|(i, (f, s))| format!("<button data-tab=\"{s}\"{}>{}</button>", if i == 0 { " class=\"on\"" } else { "" }, escape(&f.name)))
        .collect();
    let sections: String = factories.iter().zip(&slugs).enumerate().map(|(i, (f, s))| section(f, s, i == 0)).collect();
    TEMPLATE
        .replace("{{title}}", &escape(title))
        .replace("{{tabs}}", &tabs)
        .replace("{{sections}}", &sections)
}

const TEMPLATE: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{title}} · Factoryfile</title>
<style>
:root{--bg:#f7fbfb;--surface:#fff;--ink:#1f2a3a;--body:#5d6a7c;--line:#e2ebee;--a:#0d9488;--b:#4f6bff;--loop:#8b5cf6;--warn:#dc2626}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Avenir Next","Helvetica Neue",Helvetica,Arial,sans-serif;line-height:1.5}
code,pre{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12px}
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:14px 24px;background:rgba(247,251,251,.85);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
.top h1{font-size:16px;margin:0;letter-spacing:-.01em}
.top h1 span{background:linear-gradient(90deg,var(--a),var(--b));-webkit-background-clip:text;background-clip:text;color:transparent}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tabs button{border:1px solid var(--line);background:var(--surface);color:var(--body);border-radius:999px;padding:6px 14px;font:inherit;font-size:13px;cursor:pointer}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.factory{padding:24px}
.factory header h2{margin:0;font-size:28px;letter-spacing:-.02em}
.lede{margin:4px 0 0;color:var(--body);max-width:760px}
.stats{margin:6px 0 0;color:var(--body);font-size:13px}
.stage-area{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:20px;margin-top:20px}
.canvas{align-self:start;background:var(--surface);border:1px solid var(--line);border-radius:18px;overflow:auto;box-shadow:0 10px 30px rgba(15,42,51,.06)}
.graph{display:block;margin:0 auto;max-width:100%;height:auto}
.edge{fill:none;stroke:#94a3b8;stroke-width:1.6}
.edge.route{stroke:var(--a)}
.fallback{fill:none;stroke:#94a3b8;stroke-width:1.4;stroke-dasharray:4 4}
.edge-label{font-size:11px;fill:var(--body);paint-order:stroke;stroke:#fff;stroke-width:4px}
.back.bounded path{stroke:var(--loop)}
.back.bounded circle{stroke:var(--loop)}
.back.bounded text{fill:var(--loop)}
.loop path{fill:none;stroke:var(--loop);stroke-width:1.6;stroke-dasharray:5 4}
.loop,.back{cursor:pointer}
.loop circle,.back circle{fill:#fff;stroke:var(--loop);stroke-width:1.6}
.loop text,.back text{fill:var(--loop);font-size:12px;font-weight:700}
.back path{fill:none;stroke:var(--warn);stroke-width:1.6;stroke-dasharray:6 5}
.back circle,.loop.open circle{stroke:var(--warn)}
.back text,.loop.open text{fill:var(--warn)}
.loop.open path{stroke:var(--warn)}
.loop:hover path{stroke-width:3}
.legend{list-style:none;margin:0;padding:14px 18px;border-top:1px solid var(--line);display:grid;gap:8px;font-size:13px;color:var(--body)}
.legend li{cursor:pointer}
.legend li:hover b{color:var(--b)}
.legend i{display:inline-grid;place-items:center;width:20px;height:20px;margin-right:8px;border-radius:50%;border:1.6px solid var(--loop);color:var(--loop);font-style:normal;font-weight:700;font-size:11px}
.legend b{color:var(--ink)}
.node{cursor:pointer;outline:none}
.node rect{fill:#fff;stroke:var(--line);stroke-width:1.5;transition:stroke .2s}
.node.entry rect{stroke:var(--a);stroke-width:2.5}
.node:hover rect,.node:focus rect,.node.sel rect{stroke:var(--b);stroke-width:2.5}
.node .name{font-weight:700;font-size:14px;fill:var(--ink)}
.node .roles{font-size:12px;fill:var(--a)}
.node .meta{font-size:11px;fill:var(--body)}
aside{position:sticky;top:76px;align-self:start;background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:18px;max-height:calc(100vh - 96px);overflow:auto}
.hint{color:var(--body);font-size:14px}
.detail h3{margin:0 0 8px;font-size:20px}
.detail h3 small{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--a)}
.detail p{color:var(--body);margin:0 0 10px}
.row{display:flex;gap:10px;margin:6px 0;font-size:13px}
.row>span{width:72px;flex:none;color:var(--body)}
.row code{background:#eef6f6;border-radius:6px;padding:1px 6px}
.none{color:#a0aab8}
.warn{color:var(--warn);font-weight:600}
pre{white-space:pre-wrap;background:#0f2a33;color:#d7f5f0;border-radius:10px;padding:10px 12px;margin:8px 0}
.agent{border-top:1px solid var(--line);padding:12px 0 4px;margin-top:12px}
.agent b{display:block}
.agent>span{font-size:12px;color:var(--body)}
@media (max-width:900px){.stage-area{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="top"><h1><span>Factoryfile</span> · {{title}}</h1><nav class="tabs">{{tabs}}</nav></div>
{{sections}}
<script>
document.querySelectorAll(".tabs button").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll(".tabs button").forEach(x=>x.classList.toggle("on",x===b));
  document.querySelectorAll(".factory").forEach(s=>s.hidden=s.id!==b.dataset.tab);
}));
document.querySelectorAll("[data-target]").forEach(el=>{
  const show=()=>{
    const section=el.closest(".factory");
    section.querySelectorAll(".detail").forEach(d=>d.hidden=d.id!==el.dataset.target);
    section.querySelector(".hint").hidden=true;
    section.querySelectorAll(".node").forEach(n=>n.classList.toggle("sel",n===el));
  };
  el.addEventListener("click",show);
  el.addEventListener("keydown",e=>{if(e.key==="Enter")show();});
});
</script>
</body>
</html>
"##;
