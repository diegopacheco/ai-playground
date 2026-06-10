use crate::agent::Agent;

#[derive(Clone, Copy)]
pub enum Color {
    Def,
    Rgb(u8, u8, u8),
}

#[derive(Clone)]
pub struct Span {
    pub text: String,
    pub fg: Color,
    pub bold: bool,
    pub dim: bool,
}

pub type Line = Vec<Span>;

#[derive(Clone, Copy, PartialEq)]
pub enum Sort {
    Cpu,
    Ctx,
    Up,
    Status,
    Pid,
}

pub struct State {
    pub sort: Sort,
    pub interval_ms: u64,
}

const BORDER: Color = Color::Rgb(69, 71, 90);
const DIM: Color = Color::Rgb(108, 112, 134);
const TEXT: Color = Color::Rgb(205, 214, 244);
const MAUVE: Color = Color::Rgb(203, 166, 247);
const BLUE: Color = Color::Rgb(137, 180, 250);
const LAV: Color = Color::Rgb(180, 190, 254);
const GREEN: Color = Color::Rgb(166, 227, 161);
const YELLOW: Color = Color::Rgb(249, 226, 175);
const RED: Color = Color::Rgb(243, 139, 168);
const SKY: Color = Color::Rgb(137, 220, 235);
const PINK: Color = Color::Rgb(245, 194, 231);
const TEAL: Color = Color::Rgb(148, 226, 213);
const PEACH: Color = Color::Rgb(250, 179, 135);
const CTX_MAX: i64 = 200_000;

const W_PID: usize = 6;
const W_ST: usize = 6;
const W_MODEL: usize = 10;
const W_BRANCH: usize = 12;
const W_UP: usize = 7;
const W_CPU: usize = 6;
const W_MEM: usize = 6;
const W_TOK: usize = 6;
const W_BAR: usize = 10;
const W_PCT: usize = 4;
const W_MSG: usize = 5;
const W_OUT: usize = 7;

fn sp(text: impl Into<String>, fg: Color) -> Span {
    Span { text: text.into(), fg, bold: false, dim: false }
}
fn spb(text: impl Into<String>, fg: Color) -> Span {
    Span { text: text.into(), fg, bold: true, dim: false }
}
fn spd(text: impl Into<String>, fg: Color) -> Span {
    Span { text: text.into(), fg, bold: false, dim: true }
}

fn fit(s: &str, w: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > w {
        if w == 0 {
            String::new()
        } else if w == 1 {
            "…".to_string()
        } else {
            let mut t: String = chars[..w - 1].iter().collect();
            t.push('…');
            t
        }
    } else {
        let mut t = s.to_string();
        for _ in 0..(w - chars.len()) {
            t.push(' ');
        }
        t
    }
}

fn rfit(s: &str, w: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > w {
        fit(s, w)
    } else {
        let mut t = String::new();
        for _ in 0..(w - chars.len()) {
            t.push(' ');
        }
        t.push_str(s);
        t
    }
}

fn k(n: i64) -> String {
    if n < 0 {
        "-".to_string()
    } else if n < 1000 {
        n.to_string()
    } else if n < 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else if n < 1_000_000 {
        format!("{}k", n / 1000)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}

fn uptime(now: i64, start: i64) -> String {
    if start <= 0 {
        return "-".to_string();
    }
    let s = ((now - start) / 1000).max(0);
    if s < 60 {
        format!("{}s", s)
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else if s < 86400 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else {
        format!("{}d{:02}h", s / 86400, (s % 86400) / 3600)
    }
}

fn short_model(m: &str) -> String {
    let m = m.strip_prefix("claude-").unwrap_or(m);
    if m.is_empty() {
        "-".to_string()
    } else {
        m.to_string()
    }
}

fn pct_color(pct: f64) -> Color {
    if pct < 60.0 {
        GREEN
    } else if pct < 85.0 {
        YELLOW
    } else {
        RED
    }
}

fn cpu_color(cpu: f64) -> Color {
    if cpu < 30.0 {
        GREEN
    } else if cpu < 70.0 {
        YELLOW
    } else {
        RED
    }
}

fn vis_len(spans: &[Span]) -> usize {
    spans.iter().map(|s| s.text.chars().count()).sum()
}

fn line_fill(mut left: Line, right: Line, fillc: char, fill_fg: Color, w: usize) -> Line {
    let used = vis_len(&left) + vis_len(&right);
    if w > used {
        left.push(sp(fillc.to_string().repeat(w - used), fill_fg));
    }
    left.extend(right);
    left
}

fn project_width(w: usize) -> usize {
    if w >= 107 {
        w - 101
    } else {
        6
    }
}

pub fn sort_agents(agents: &mut [Agent], sort: Sort) {
    match sort {
        Sort::Cpu => agents.sort_by(|a, b| b.cpu.partial_cmp(&a.cpu).unwrap()),
        Sort::Ctx => agents.sort_by(|a, b| b.ctx_tokens.cmp(&a.ctx_tokens)),
        Sort::Up => agents.sort_by(|a, b| a.started_at.cmp(&b.started_at)),
        Sort::Pid => agents.sort_by(|a, b| a.pid.cmp(&b.pid)),
        Sort::Status => agents.sort_by(|a, b| {
            rank(&a.status)
                .cmp(&rank(&b.status))
                .then(b.cpu.partial_cmp(&a.cpu).unwrap())
        }),
    }
}

fn rank(status: &str) -> u8 {
    match status {
        "busy" => 0,
        "idle" => 1,
        _ => 2,
    }
}

fn sort_label(sort: Sort) -> &'static str {
    match sort {
        Sort::Cpu => "cpu",
        Sort::Ctx => "ctx",
        Sort::Up => "uptime",
        Sort::Status => "status",
        Sort::Pid => "pid",
    }
}

pub fn frame(
    agents: &[Agent],
    state: &State,
    w: usize,
    h: usize,
    now_ms: i64,
    clock: &str,
    host: &str,
) -> Vec<Line> {
    let w = w.max(60);
    let p = project_width(w);
    let mut out: Vec<Line> = Vec::new();

    out.push(line_fill(
        vec![sp("┌─ ", BORDER), spb("AGENTOP", MAUVE), sp("  ", Color::Def), spd("top for claude code agents", DIM)],
        vec![sp("─ ", BORDER), sp(clock, BLUE), sp(" ─┐", BORDER)],
        '─',
        BORDER,
        w,
    ));

    let busy = agents.iter().filter(|a| a.status == "busy").count();
    let idle = agents.iter().filter(|a| a.status == "idle").count();
    let total_ctx: i64 = agents.iter().map(|a| a.ctx_tokens).sum();
    let total_cpu: f64 = agents.iter().map(|a| a.cpu).sum();
    let version = agents.first().map(|a| a.version.clone()).unwrap_or_default();
    out.push(line_fill(
        vec![
            sp("│ ", BORDER),
            spd("RUNNING ", DIM),
            spb(agents.len().to_string(), LAV),
            spd("   BUSY ", DIM),
            spb(busy.to_string(), GREEN),
            spd("   IDLE ", DIM),
            spb(idle.to_string(), SKY),
            spd("   CTX ", DIM),
            spb(k(total_ctx), PINK),
            spd("   CPU ", DIM),
            spb(format!("{:.0}%", total_cpu), TEAL),
            spd("   CC ", DIM),
            sp(version, TEAL),
            spd("   HOST ", DIM),
            sp(host, LAV),
        ],
        vec![sp(" │", BORDER)],
        ' ',
        Color::Def,
        w,
    ));

    out.push(line_fill(vec![sp("├", BORDER)], vec![sp("┤", BORDER)], '─', BORDER, w));

    let mut header = vec![sp("│ ", BORDER)];
    push_cells(
        &mut header,
        vec![
            spb(rfit("PID", W_PID), LAV),
            spb(fit("ST", W_ST), LAV),
            spb(fit("MODEL", W_MODEL), LAV),
            spb(fit("PROJECT", p), LAV),
            spb(fit("BRANCH", W_BRANCH), LAV),
            spb(rfit("UP", W_UP), LAV),
            spb(rfit("CPU%", W_CPU), LAV),
            spb(rfit("MEM%", W_MEM), LAV),
            spb(rfit("CTX", W_TOK), LAV),
            spb(fit("CONTEXT", W_BAR), LAV),
            spb(rfit("%", W_PCT), LAV),
            spb(rfit("MSG", W_MSG), LAV),
            spb(rfit("OUT", W_OUT), LAV),
        ],
    );
    out.push(line_fill(header, vec![sp(" │", BORDER)], ' ', Color::Def, w));

    let area = if h > 0 { h.saturating_sub(5) } else { agents.len() };
    let mut rendered = 0usize;

    if agents.is_empty() {
        out.push(center_row("no running claude code agents found", DIM, w));
        rendered += 1;
    } else {
        let overflow = h > 0 && agents.len() > area;
        let show = if overflow { area.saturating_sub(1) } else { agents.len() };
        for a in agents.iter().take(show) {
            out.push(agent_row(a, p, now_ms, w));
            rendered += 1;
        }
        if overflow {
            out.push(center_row(&format!("… {} more agents", agents.len() - show), DIM, w));
            rendered += 1;
        }
    }

    if h > 0 {
        while rendered < area {
            out.push(line_fill(vec![sp("│", BORDER)], vec![sp("│", BORDER)], ' ', Color::Def, w));
            rendered += 1;
        }
    }

    out.push(line_fill(
        vec![
            sp("└─ ", BORDER),
            sp("q", PEACH),
            spd(" quit   ", DIM),
            sp("s", PEACH),
            spd(" sort   ", DIM),
            sp("+/-", PEACH),
            spd(" speed   ", DIM),
            spd("sort:", DIM),
            sp(sort_label(state.sort), LAV),
            spd("   every ", DIM),
            sp(format!("{}s", state.interval_ms / 1000), LAV),
        ],
        vec![sp(" ─┘", BORDER)],
        '─',
        BORDER,
        w,
    ));

    out
}

fn push_cells(line: &mut Line, cells: Vec<Span>) {
    let n = cells.len();
    for (i, c) in cells.into_iter().enumerate() {
        line.push(c);
        if i + 1 < n {
            line.push(sp(" ", Color::Def));
        }
    }
}

fn agent_row(a: &Agent, p: usize, now_ms: i64, w: usize) -> Line {
    let (stcol, sttext) = match a.status.as_str() {
        "busy" => (GREEN, "BUSY"),
        "idle" => (SKY, "IDLE"),
        _ => (DIM, "?"),
    };
    let pct = if CTX_MAX > 0 {
        (a.ctx_tokens as f64 / CTX_MAX as f64 * 100.0).min(100.0)
    } else {
        0.0
    };
    let filled = ((pct / 100.0) * W_BAR as f64).round() as usize;
    let filled = filled.min(W_BAR);
    let gcol = pct_color(pct);
    let branch = if a.branch.is_empty() { "-" } else { &a.branch };

    let mut row = vec![sp("│ ", BORDER)];
    row.push(sp(rfit(&a.pid.to_string(), W_PID), LAV));
    row.push(sp(" ", Color::Def));
    row.push(sp("●", stcol));
    row.push(sp(" ", Color::Def));
    row.push(sp(fit(sttext, W_ST - 2), stcol));
    row.push(sp(" ", Color::Def));
    row.push(sp(fit(&short_model(&a.model), W_MODEL), TEAL));
    row.push(sp(" ", Color::Def));
    row.push(sp(fit(&a.project, p), TEXT));
    row.push(sp(" ", Color::Def));
    row.push(spd(fit(branch, W_BRANCH), DIM));
    row.push(sp(" ", Color::Def));
    row.push(sp(rfit(&uptime(now_ms, a.started_at), W_UP), TEXT));
    row.push(sp(" ", Color::Def));
    row.push(sp(rfit(&format!("{:.1}", a.cpu), W_CPU), cpu_color(a.cpu)));
    row.push(sp(" ", Color::Def));
    row.push(sp(rfit(&format!("{:.1}", a.mem), W_MEM), PEACH));
    row.push(sp(" ", Color::Def));
    row.push(sp(rfit(&k(a.ctx_tokens), W_TOK), PINK));
    row.push(sp(" ", Color::Def));
    row.push(sp("█".repeat(filled), gcol));
    row.push(sp("░".repeat(W_BAR - filled), BORDER));
    row.push(sp(" ", Color::Def));
    row.push(sp(rfit(&format!("{:.0}%", pct), W_PCT), gcol));
    row.push(sp(" ", Color::Def));
    row.push(sp(rfit(&a.msgs.to_string(), W_MSG), LAV));
    row.push(sp(" ", Color::Def));
    row.push(sp(rfit(&k(a.out_tokens), W_OUT), MAUVE));

    line_fill(row, vec![sp(" │", BORDER)], ' ', Color::Def, w)
}

fn center_row(text: &str, color: Color, w: usize) -> Line {
    let inner = w.saturating_sub(2);
    let tlen = text.chars().count().min(inner);
    let pad = (inner - tlen) / 2;
    line_fill(
        vec![sp("│", BORDER), sp(" ".repeat(pad), Color::Def), sp(text, color)],
        vec![sp("│", BORDER)],
        ' ',
        Color::Def,
        w,
    )
}

pub fn html(frame: &[Line]) -> String {
    let mut body = String::new();
    for line in frame {
        body.push_str("<div class=\"l\">");
        for span in line {
            let mut style = String::new();
            if let Color::Rgb(r, g, b) = span.fg {
                style.push_str(&format!("color:rgb({},{},{});", r, g, b));
            }
            if span.bold {
                style.push_str("font-weight:700;");
            }
            if span.dim {
                style.push_str("opacity:.62;");
            }
            body.push_str(&format!("<span style=\"{}\">{}</span>", style, esc(&span.text)));
        }
        body.push_str("</div>");
    }
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>\
        body{{margin:0;background:#181825;display:flex;justify-content:center;align-items:center;padding:42px;font-family:'JetBrains Mono','SF Mono',Menlo,monospace}}\
        .win{{background:#1e1e2e;border-radius:12px;box-shadow:0 30px 80px rgba(0,0,0,.55);overflow:hidden;border:1px solid #313244}}\
        .bar{{height:38px;background:#11111b;display:flex;align-items:center;padding:0 16px;gap:8px}}\
        .dot{{width:12px;height:12px;border-radius:50%}}\
        .r{{background:#f38ba8}}.y{{background:#f9e2af}}.g{{background:#a6e3a1}}\
        .t{{margin-left:12px;color:#6c7086;font-size:13px}}\
        .scr{{padding:18px 20px;font-size:14px;line-height:1.45}}\
        .l{{white-space:pre}}\
        </style></head><body><div class=\"win\"><div class=\"bar\"><div class=\"dot r\"></div>\
        <div class=\"dot y\"></div><div class=\"dot g\"></div><div class=\"t\">agentop</div></div>\
        <div class=\"scr\">{}</div></div></body></html>",
        body
    )
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}
