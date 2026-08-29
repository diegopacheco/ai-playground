use crate::monitor::{Monitor, Resolution};

const LABEL_WIDTH: usize = 18;

pub fn text(monitors: &[Monitor]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n🖥️  monitor-stats — {}\n\n",
        headline(monitors.len())
    ));
    for (index, monitor) in monitors.iter().enumerate() {
        out.push_str(&card(index + 1, monitor));
    }
    out
}

fn headline(count: usize) -> String {
    match count {
        0 => "no monitors detected".to_string(),
        1 => "1 monitor detected".to_string(),
        n => format!("{n} monitors detected"),
    }
}

fn card(position: usize, monitor: &Monitor) -> String {
    let icon = if monitor.builtin { "💻" } else { "🖥️ " };
    let main = if monitor.main { "  ⭐ main" } else { "" };
    let mut out = format!("  {position}. {icon} {}{main}\n", monitor.name);
    out.push_str(&row("🔖", "Brand", &monitor.brand));
    out.push_str(&row("📏", "Size", &size(monitor.diagonal_inches)));
    out.push_str(&row(
        "🧮",
        "Max resolution",
        &resolution(&monitor.max_resolution),
    ));
    out.push_str(&row(
        "🔄",
        "Max refresh rate",
        &refresh(monitor.max_refresh_hz),
    ));
    out.push('\n');
    out
}

fn row(icon: &str, label: &str, value: &str) -> String {
    format!("     {icon}  {label:<LABEL_WIDTH$}{value}\n")
}

pub fn size(inches: f64) -> String {
    if inches <= 0.0 {
        return "unknown".to_string();
    }
    format!("{inches:.1}\"")
}

pub fn resolution(resolution: &Option<Resolution>) -> String {
    match resolution {
        None => "unknown".to_string(),
        Some(res) => format!("{} × {}", res.width, res.height),
    }
}

pub fn refresh(hz: Option<f64>) -> String {
    match hz {
        None => "unknown".to_string(),
        Some(hz) if (hz - hz.round()).abs() < 0.05 => format!("{} Hz", hz.round() as u64),
        Some(hz) => format!("{hz:.1} Hz"),
    }
}

pub fn json(monitors: &[Monitor]) -> String {
    let entries: Vec<String> = monitors.iter().map(entry).collect();
    format!("[\n{}\n]\n", entries.join(",\n"))
}

fn entry(monitor: &Monitor) -> String {
    let (width, height, mode_hz) = match &monitor.max_resolution {
        Some(res) => (
            res.width.to_string(),
            res.height.to_string(),
            number(Some(res.refresh_hz)),
        ),
        None => ("null".to_string(), "null".to_string(), "null".to_string()),
    };
    format!(
        "  {{\n    \"name\": {},\n    \"brand\": {},\n    \"builtin\": {},\n    \"main\": {},\n    \"diagonal_inches\": {},\n    \"max_width\": {width},\n    \"max_height\": {height},\n    \"max_resolution_refresh_hz\": {mode_hz},\n    \"max_refresh_hz\": {}\n  }}",
        quote(&monitor.name),
        quote(&monitor.brand),
        monitor.builtin,
        monitor.main,
        number((monitor.diagonal_inches > 0.0).then_some(monitor.diagonal_inches)),
        number(monitor.max_refresh_hz),
    )
}

fn number(value: Option<f64>) -> String {
    match value {
        None => "null".to_string(),
        Some(value) => format!("{}", (value * 100.0).round() / 100.0),
    }
}

fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> Monitor {
        Monitor {
            name: "LS32A70".to_string(),
            brand: "Samsung".to_string(),
            builtin: false,
            main: true,
            diagonal_inches: 31.8,
            max_resolution: Some(Resolution {
                width: 3840,
                height: 2160,
                refresh_hz: 30.0,
            }),
            max_refresh_hz: Some(75.0),
        }
    }

    #[test]
    fn drops_the_decimal_noise_from_a_whole_refresh_rate() {
        assert_eq!(refresh(Some(120.0)), "120 Hz");
        assert_eq!(refresh(Some(59.93998718261719)), "59.9 Hz");
    }

    #[test]
    fn says_unknown_instead_of_inventing_a_zero_the_user_would_trust() {
        assert_eq!(refresh(None), "unknown");
        assert_eq!(resolution(&None), "unknown");
        assert_eq!(size(0.0), "unknown");
    }

    #[test]
    fn counts_monitors_in_the_headline_with_matching_grammar() {
        assert_eq!(headline(0), "no monitors detected");
        assert_eq!(headline(1), "1 monitor detected");
        assert_eq!(headline(3), "3 monitors detected");
    }

    #[test]
    fn shows_every_requested_field_for_each_monitor() {
        let out = text(&[monitor()]);
        assert!(out.contains("LS32A70"));
        assert!(out.contains("Samsung"));
        assert!(out.contains("31.8\""));
        assert!(out.contains("3840 × 2160"));
        assert!(out.contains("75 Hz"));
    }

    #[test]
    fn marks_the_built_in_panel_apart_from_an_external_one() {
        let mut builtin = monitor();
        builtin.builtin = true;
        assert!(text(&[builtin]).contains("💻"));
        assert!(text(&[monitor()]).contains("🖥️"));
    }

    #[test]
    fn emits_json_that_keeps_null_apart_from_a_measured_zero() {
        let mut blind = monitor();
        blind.diagonal_inches = 0.0;
        blind.max_refresh_hz = None;
        blind.max_resolution = None;
        let out = json(&[blind]);
        assert!(out.contains("\"diagonal_inches\": null"));
        assert!(out.contains("\"max_refresh_hz\": null"));
        assert!(out.contains("\"max_width\": null"));
    }

    #[test]
    fn escapes_a_panel_name_so_the_json_stays_parsable() {
        let mut odd = monitor();
        odd.name = "22\" \\ \n".to_string();
        assert!(json(&[odd]).contains(r#""name": "22\" \\ \n""#));
    }

    #[test]
    fn emits_an_empty_json_array_when_nothing_is_attached() {
        assert_eq!(json(&[]).trim(), "[\n\n]");
    }
}
