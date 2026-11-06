use factoryfile::{Factory, Loop, Stage};

use crate::layout::{Edge, Layout, count, layout, route_text, work};

const GAP: &str = "   ";

struct Paint {
    on: bool,
}

impl Paint {
    fn wrap(&self, code: &str, text: &str) -> String {
        if self.on { format!("\x1b[{code}m{text}\x1b[0m") } else { text.to_string() }
    }

    fn bold(&self, t: &str) -> String {
        self.wrap("1", t)
    }

    fn dim(&self, t: &str) -> String {
        self.wrap("2", t)
    }

    fn teal(&self, t: &str) -> String {
        self.wrap("36", t)
    }

    fn violet(&self, t: &str) -> String {
        self.wrap("35", t)
    }

    fn red(&self, t: &str) -> String {
        self.wrap("31", t)
    }

    fn green(&self, t: &str) -> String {
        self.wrap("32", t)
    }
}

struct Line {
    text: String,
    width: usize,
}

fn width(s: &str) -> usize {
    s.chars().count()
}

fn pad(line: &Line, to: usize) -> String {
    format!("{}{}", line.text, " ".repeat(to.saturating_sub(line.width)))
}

pub fn render(factory: &Factory, color: bool) -> String {
    let p = Paint { on: color };
    let graph = layout(factory);
    let mut out = header(factory, &p);
    for (i, layer) in graph.layers.iter().enumerate() {
        if i > 0 {
            out.push_str(&format!("{}\n", p.dim("        │")));
            out.push_str(&format!("{}\n", p.dim("        ▼")));
        }
        out.push_str(&render_layer(factory, &graph, layer, i + 1, &p));
    }
    out.push_str(&render_loops(factory, &graph, &p));
    out
}

fn header(factory: &Factory, p: &Paint) -> String {
    let mut out = format!("{} {}\n", p.violet("factory"), p.bold(&factory.name));
    if let Some(d) = &factory.description {
        out.push_str(&format!("{}\n", p.dim(d)));
    }
    let entry = factory.entry.as_deref().unwrap_or("none");
    out.push_str(&format!(
        "{}\n\n",
        p.dim(&format!(
            "entry {entry} · {} · {} · {}",
            count(factory.stages.len(), "stage"),
            count(factory.agents.len(), "agent"),
            count(factory.loops.len(), "loop")
        ))
    ));
    out
}

fn render_layer(factory: &Factory, graph: &Layout, layer: &[usize], number: usize, p: &Paint) -> String {
    let boxes: Vec<Vec<Line>> = layer.iter().map(|s| stage_box(factory, graph, &factory.stages[*s], p)).collect();
    let height = boxes.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = boxes.iter().map(|b| b.iter().map(|l| l.width).max().unwrap_or(0)).collect();
    let mut out = String::new();
    for row in 0..height {
        let label = if row == 1 { format!("L{number:<3}") } else { "    ".to_string() };
        let cells: Vec<String> = boxes
            .iter()
            .zip(&widths)
            .map(|(b, w)| b.get(row).map_or(" ".repeat(*w), |l| pad(l, *w)))
            .collect();
        out.push_str(&format!("{}  {}\n", p.dim(&label), cells.join(GAP).trim_end()));
    }
    out
}

fn stage_box(factory: &Factory, graph: &Layout, stage: &Stage, p: &Paint) -> Vec<Line> {
    let is_entry = factory.entry.as_deref() == Some(stage.name.as_str());
    let title = if is_entry { format!("▶ {}", stage.name) } else { stage.name.clone() };
    let agents = work(stage);
    let mut rows: Vec<(String, String)> = vec![(title.clone(), p.bold(&title)), (agents.clone(), p.teal(&agents))];
    for artifact in &stage.artifacts {
        let text = format!("▸ {artifact}");
        rows.push((text.clone(), p.dim(&text)));
    }
    let inner = rows.iter().map(|(plain, _)| width(plain)).max().unwrap_or(0);
    let mut lines = vec![Line { text: format!("┌{}┐", "─".repeat(inner + 2)), width: inner + 4 }];
    for (plain, painted) in &rows {
        let fill = " ".repeat(inner - width(plain));
        lines.push(Line { text: format!("│ {painted}{fill} │"), width: inner + 4 });
    }
    lines.push(Line { text: format!("└{}┘", "─".repeat(inner + 2)), width: inner + 4 });
    lines.extend(edges_below(factory, graph, stage, p));
    lines
}

fn edges_below<'a>(factory: &Factory, graph: &'a Layout, stage: &Stage, p: &Paint) -> Vec<Line> {
    let me = factory.stages.iter().position(|s| s.name == stage.name).unwrap_or(0);
    let mut lines = Vec::new();
    let mine = |edges: &'a [Edge]| edges.iter().filter(move |e| e.from == me);
    let forward: Vec<&str> = mine(&graph.forward).filter(|e| e.route.is_none()).map(|e| factory.stages[e.to].name.as_str()).collect();
    if !forward.is_empty() {
        let text = format!("  └▶ {}", forward.join(", "));
        lines.push(Line { width: width(&text), text: p.green(&text) });
    }
    for e in mine(&graph.forward).filter(|e| e.route.is_some()) {
        let text = format!("  └▶ {}  on {}", factory.stages[e.to].name, route_text(stage, e).unwrap_or_default());
        lines.push(Line { width: width(&text), text: p.green(&text) });
    }
    for e in mine(&graph.backward) {
        let on = route_text(stage, e).map(|r| format!("  on {r}")).unwrap_or_default();
        let text = format!("  ↩ {}{on}{}", factory.stages[e.to].name, if e.bounded { "" } else { " (unbounded)" });
        lines.push(Line { width: width(&text), text: if e.bounded { p.violet(&text) } else { p.red(&text) } });
    }
    for route in &stage.routes {
        for fallback in route.fallbacks() {
            let label = fallback.label.as_deref().map(|l| format!(" · {l}")).unwrap_or_default();
            let text = format!("  ┄▶ {}  fallback of {}{label}", fallback.to, route.outcome);
            lines.push(Line { width: width(&text), text: p.dim(&text) });
        }
    }
    if stage.next.is_empty() && stage.routes.is_empty() && !factory.loops.iter().any(|l| l.from == stage.name) {
        let text = stage.end.as_deref().map_or("  ■ end".to_string(), |end| format!("  ■ end {end}"));
        lines.push(Line { width: width(&text), text: p.dim(&text) });
    }
    lines
}

fn render_loops(factory: &Factory, graph: &Layout, p: &Paint) -> String {
    let unbounded: Vec<&Edge> = graph.backward.iter().filter(|e| !e.bounded).collect();
    if factory.loops.is_empty() && unbounded.is_empty() {
        return String::new();
    }
    let mut out = format!("\n{}\n", p.violet("loops"));
    let name_width = factory.loops.iter().map(|l| width(&l.name)).max().unwrap_or(0);
    for l in &factory.loops {
        out.push_str(&format!("  {}\n", loop_line(l, name_width, p)));
    }
    for e in unbounded {
        let from = &factory.stages[e.from];
        let via = route_text(from, e).map_or("next".to_string(), |r| format!("on {r}"));
        let text = format!("↩ {via} {} ──▶ {}  no max, it may never end", from.name, factory.stages[e.to].name);
        out.push_str(&format!("  {}\n", p.red(&text)));
    }
    out
}

fn loop_line(l: &Loop, name_width: usize, p: &Paint) -> String {
    let edge = format!("{} ──▶ {}", l.from, l.to);
    let until = l.until.as_deref().map(|u| format!("  until {u}")).unwrap_or_default();
    let max = match l.max {
        Some(m) => p.dim(&format!("  max {m}")),
        None => p.red("  no max, it may never end"),
    };
    format!("{} {:<name_width$}  {}{}{}", p.violet("↺"), l.name, p.teal(&edge), p.dim(&until), max)
}
