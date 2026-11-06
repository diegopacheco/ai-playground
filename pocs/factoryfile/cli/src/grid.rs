use factoryfile::Factory;

use crate::html::{H, Placed, W, escape};
use crate::layout::{Layout, route_text};

const GAP: i64 = 180;
const PAD: i64 = 28;
const TRACK: i64 = 26;
const MIN_CHANNEL: i64 = 76;
const PORT: i64 = 28;
const PAIR: i64 = 12;
const CORNER: i64 = 10;

type Point = (i64, i64);
type Rect = (i64, i64, i64, i64);

pub(crate) struct Drawn {
    pub(crate) places: Vec<Placed>,
    pub(crate) edges: String,
    pub(crate) right: usize,
    pub(crate) height: usize,
}

struct Line {
    from: usize,
    to: usize,
    class: &'static str,
    label: Option<String>,
    title: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Top,
    Bottom,
}

struct Around {
    line: usize,
    channel: usize,
    exit: Side,
    enter: Side,
}

pub(crate) fn draw(factory: &Factory, graph: &Layout, cells: &[(usize, usize)]) -> Drawn {
    let lines = lines(factory, graph);
    let cell = |i: usize| (cells[i].0 as i64, cells[i].1 as i64);
    let rows = cells.iter().map(|(_, r)| r + 1).max().unwrap_or(0);
    let cols = cells.iter().map(|(c, _)| c + 1).max().unwrap_or(0) as i64;
    let adjacent = |l: &Line| {
        let ((ac, ar), (bc, br)) = (cell(l.from), cell(l.to));
        (ac == bc && ar.abs_diff(br) == 1) || (ar == br && ac.abs_diff(bc) == 1)
    };
    let mut tracks: Vec<Vec<usize>> = vec![Vec::new(); rows + 1];
    let mut arounds = Vec::new();
    for (i, l) in lines.iter().enumerate().filter(|(_, l)| !adjacent(l)) {
        let ((ac, ar), (bc, br)) = (cell(l.from), cell(l.to));
        let (channel, exit, enter) = if br > ar {
            (ar + 1, Side::Bottom, Side::Top)
        } else if br < ar {
            (ar, Side::Top, Side::Bottom)
        } else if bc > ac {
            (ar + 1, Side::Bottom, Side::Bottom)
        } else {
            (ar, Side::Top, Side::Top)
        };
        tracks[channel as usize].push(i);
        arounds.push(Around { line: i, channel: channel as usize, exit, enter });
    }
    let heights: Vec<i64> = tracks
        .iter()
        .enumerate()
        .map(|(c, t)| {
            let n = t.len() as i64;
            match (c == 0 || c == rows, n) {
                (true, 0) => PAD,
                (true, _) => n * TRACK + 2 * PAD,
                (false, _) => MIN_CHANNEL.max(n * TRACK + 2 * PAD),
            }
        })
        .collect();
    let mut row_y = Vec::new();
    let mut y = heights[0];
    for h in heights.iter().skip(1) {
        row_y.push(y);
        y += H as i64 + h;
    }
    let height = y;
    let places: Vec<Placed> = cells
        .iter()
        .map(|(c, r)| Placed { x: (PAD + *c as i64 * (W as i64 + GAP)) as usize, y: row_y[*r] as usize, layer: *r })
        .collect();
    let node = |i: usize| (places[i].x as i64, places[i].y as i64);
    let track_y = |channel: usize, k: usize| {
        let top = if channel == 0 { 0 } else { row_y[channel - 1] + H as i64 };
        let n = tracks[channel].len() as i64;
        top + (heights[channel] - (n - 1) * TRACK) / 2 + k as i64 * TRACK
    };
    let mut ports: Vec<(i64, usize, bool)> = Vec::new();
    let mut port = |stage: usize, channel: usize, other: usize, leaving: bool| {
        let (x, _) = node(stage);
        let (ox, _) = node(other);
        let left = ox < x || (ox == x && leaving);
        let k = ports.iter().filter(|p| **p == (x, channel, left)).count() as i64;
        ports.push((x, channel, left));
        if left { x + PORT * (k + 1) } else { x + W as i64 - PORT * (k + 1) }
    };
    let side_y = |stage: usize, side: Side| if side == Side::Top { node(stage).1 } else { node(stage).1 + H as i64 };
    let mut paths: Vec<Vec<Point>> = vec![Vec::new(); lines.len()];
    for a in &arounds {
        let l = &lines[a.line];
        let k = tracks[a.channel].iter().position(|i| *i == a.line).unwrap_or(0);
        let ty = track_y(a.channel, k);
        let sx = port(l.from, a.channel, l.to, true);
        let ex = port(l.to, a.channel, l.from, false);
        paths[a.line] = vec![(sx, side_y(l.from, a.exit)), (sx, ty), (ex, ty), (ex, side_y(l.to, a.enter))];
    }
    for (i, l) in lines.iter().enumerate().filter(|(_, l)| adjacent(l)) {
        let paired = lines.iter().any(|o| o.from == l.to && o.to == l.from && adjacent(o));
        let ((ax, ay), (bx, by)) = (node(l.from), node(l.to));
        paths[i] = if ax == bx {
            let x = ax + W as i64 / 2 + if paired { if by > ay { -PAIR } else { PAIR } } else { 0 };
            if by > ay { vec![(x, ay + H as i64), (x, by)] } else { vec![(x, ay), (x, by + H as i64)] }
        } else {
            let y = ay + H as i64 / 2 + if paired { if bx > ax { -PAIR } else { PAIR } } else { 0 };
            if bx > ax { vec![(ax + W as i64, y), (bx, y)] } else { vec![(ax, y), (bx + W as i64, y)] }
        };
    }
    let mut taken: Vec<Rect> = places.iter().map(|p| (p.x as i64, p.y as i64, p.x as i64 + W as i64, p.y as i64 + H as i64)).collect();
    let mut edges = String::new();
    for (i, (l, path)) in lines.iter().zip(&paths).enumerate() {
        let others: Vec<Rect> = paths.iter().enumerate().filter(|(j, _)| *j != i).flat_map(|(_, p)| segments(p)).collect();
        let text = l.label.as_deref().map(|t| place(t, path, &mut taken, &others)).unwrap_or_default();
        let title = l.title.as_deref().map(|t| format!("<title>{}</title>", escape(t))).unwrap_or_default();
        edges.push_str(&format!(
            "<g class=\"{}\">{title}<path d=\"{}\" marker-end=\"url(#arrow)\"/>{text}</g>",
            l.class,
            rounded(path)
        ));
    }
    let right = PAD + cols * W as i64 + (cols - 1).max(0) * GAP;
    Drawn { places, edges, right: right as usize, height: height as usize }
}

fn lines(factory: &Factory, graph: &Layout) -> Vec<Line> {
    let mut out: Vec<Line> = graph
        .forward
        .iter()
        .map(|e| {
            let label = route_text(&factory.stages[e.from], e);
            let class = if label.is_some() { "edge route" } else { "edge" };
            Line { from: e.from, to: e.to, class, label, title: None }
        })
        .collect();
    for e in &graph.backward {
        let from = &factory.stages[e.from];
        let label = route_text(from, e);
        let via = label.as_deref().map_or("next".to_string(), |r| format!("on {r}"));
        let (class, bound) = if e.bounded { ("back bounded", "bounded") } else { ("back", "unbounded") };
        let title = Some(format!("{bound} {via} {} → {}", from.name, factory.stages[e.to].name));
        out.push(Line { from: e.from, to: e.to, class, label: Some(label.unwrap_or_else(|| "next".to_string())), title });
    }
    for (i, stage) in factory.stages.iter().enumerate() {
        for f in stage.routes.iter().flat_map(|r| r.fallbacks()) {
            if let Some(to) = factory.stages.iter().position(|s| s.name == f.to) {
                let label = Some(f.label.clone().unwrap_or_else(|| "exhausted".to_string()));
                out.push(Line { from: i, to, class: "fallback", label, title: None });
            }
        }
    }
    out
}

fn rounded(points: &[Point]) -> String {
    let mut d = format!("M{} {}", points[0].0, points[0].1);
    for w in points.windows(3) {
        let (a, b, c) = (w[0], w[1], w[2]);
        let r = CORNER.min(span(a, b) / 2).min(span(b, c) / 2);
        let (p, q) = (toward(b, a, r), toward(b, c, r));
        d.push_str(&format!(" L{} {} Q{} {} {} {}", p.0, p.1, b.0, b.1, q.0, q.1));
    }
    let last = points[points.len() - 1];
    d.push_str(&format!(" L{} {}", last.0, last.1));
    d
}

fn span(a: Point, b: Point) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn toward(from: Point, to: Point, r: i64) -> Point {
    (from.0 + (to.0 - from.0).signum() * r, from.1 + (to.1 - from.1).signum() * r)
}

fn candidates(path: &[Point], w: i64) -> Vec<Point> {
    let mut out = Vec::new();
    let segments: Vec<(Point, Point)> = path.windows(2).map(|s| (s[0], s[1])).collect();
    let longest = segments.iter().copied().filter(|(a, b)| a.1 == b.1).max_by_key(|(a, b)| span(*a, *b));
    let order = longest.into_iter().chain(segments.iter().copied().filter(|(a, b)| a.0 == b.0));
    for (a, b) in order {
        for t in [50, 35, 65, 20, 80] {
            let at = (a.0 + (b.0 - a.0) * t / 100, a.1 + (b.1 - a.1) * t / 100);
            if a.1 == b.1 {
                out.extend([at, (at.0, at.1 - 12), (at.0, at.1 + 14)]);
            } else {
                out.extend([(at.0 + w / 2 + 6, at.1), (at.0 - w / 2 - 6, at.1)]);
            }
        }
    }
    out
}

fn segments(path: &[Point]) -> Vec<Rect> {
    path.windows(2).map(|s| (s[0].0.min(s[1].0) - 2, s[0].1.min(s[1].1) - 2, s[0].0.max(s[1].0) + 2, s[0].1.max(s[1].1) + 2)).collect()
}

fn place(text: &str, path: &[Point], taken: &mut Vec<Rect>, others: &[Rect]) -> String {
    let w = text.chars().count() as i64 * 6 + 10;
    let rect = |(x, y): Point| (x - w / 2, y - 8, x + w / 2, y + 8);
    let hits = |r: &Rect, t: &Rect| r.0 < t.2 && t.0 < r.2 && r.1 < t.3 && t.1 < r.3;
    let free = |r: &Rect| !taken.iter().chain(others).any(|t| hits(r, t));
    let all = candidates(path, w);
    let at = all.iter().copied().find(|p| free(&rect(*p))).unwrap_or(all[0]);
    taken.push(rect(at));
    format!("<text class=\"edge-label\" x=\"{}\" y=\"{}\" text-anchor=\"middle\">{}</text>", at.0, at.1 + 4, escape(text))
}
