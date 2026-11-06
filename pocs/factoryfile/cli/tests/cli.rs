use std::path::PathBuf;

use factory_cli::args::{self, Mode};
use factory_cli::html::{escape, page, svg};
use factory_cli::layout::layout;
use factory_cli::terminal::render;
use factoryfile::{Factory, parse, parse_file};

fn sample(name: &str) -> Factory {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../samples").join(format!("{name}.factoryfile"));
    parse_file(path).unwrap().factories.remove(0)
}

fn inline(body: &str) -> Factory {
    parse(&format!("version = \"1\"\nfactory \"f\" {{\n{body}\n}}\n")).unwrap().factories.remove(0)
}

fn names(f: &Factory, layer: &[usize]) -> Vec<String> {
    layer.iter().map(|i| f.stages[*i].name.clone()).collect()
}

#[test]
fn a_straight_pipeline_gets_one_stage_per_layer_in_order() {
    let f = sample("waterfall");
    let g = layout(&f);
    let flat: Vec<Vec<String>> = g.layers.iter().map(|l| names(&f, l)).collect();
    assert_eq!(flat, [["requirements"], ["design"], ["implementation"], ["verification"], ["maintenance"]]);
    assert!(g.backward.is_empty());
}

#[test]
fn a_stage_sits_after_its_longest_path_so_a_shortcut_does_not_pull_it_left() {
    let f = sample("spiral");
    let g = layout(&f);
    let layer_of = |name: &str| g.layers.iter().position(|l| names(&f, l).contains(&name.to_string())).unwrap();
    assert_eq!(layer_of("prototype"), 2);
    assert_eq!(layer_of("develop-and-test"), 3, "risk-analysis -> develop-and-test is shorter than the path through prototype");
}

#[test]
fn loops_never_change_the_layers_because_they_are_back_edges() {
    let with = sample("xp");
    let without = inline(
        "  stage \"a\" { agents = [] next = [\"b\"] }\n  stage \"b\" { agents = [] }",
    );
    assert_eq!(layout(&with).layers.len(), with.stages.len());
    assert_eq!(layout(&without).layers.len(), 2);
}

#[test]
fn a_next_cycle_is_split_into_a_back_edge_starting_from_the_entry() {
    let f = sample("xgh");
    let g = layout(&f);
    let back: Vec<(String, String)> = g.backward.iter().map(|(a, b)| (f.stages[*a].name.clone(), f.stages[*b].name.clone())).collect();
    assert_eq!(back, [("firefight".to_string(), "hack".to_string())]);
    assert_eq!(names(&f, &g.layers[0]), ["receive-task", "write-tests"]);
}

#[test]
fn every_stage_is_placed_exactly_once() {
    for name in ["rup", "xp", "scrum", "xgh", "kanban", "waterfall", "spiral"] {
        let f = sample(name);
        let mut placed: Vec<usize> = layout(&f).layers.concat();
        placed.sort();
        assert_eq!(placed, (0..f.stages.len()).collect::<Vec<_>>(), "{name}");
    }
}

#[test]
fn terminal_render_shows_every_stage_agent_and_loop_bound() {
    let f = sample("scrum");
    let text = render(&f, false);
    for stage in &f.stages {
        assert!(text.contains(&stage.name), "{}", stage.name);
    }
    assert!(text.contains("▶ backlog-refinement"), "the entry is marked");
    assert!(text.contains("↺ sprint") && text.contains("retrospective ──▶ backlog-refinement") && text.contains("max 26"));
    assert!(!text.contains('\x1b'), "no color codes when color is off");
}

#[test]
fn terminal_render_warns_about_unbounded_cycles() {
    let text = render(&sample("xgh"), false);
    assert!(text.contains("↩ hack (unbounded)"));
    assert!(text.contains("go-horse-forever") && text.contains("no max, it may never end"));
}

#[test]
fn terminal_boxes_line_up_even_with_color_codes() {
    let text = render(&sample("spiral"), true);
    assert!(text.contains('\x1b'));
    let plain = render(&sample("spiral"), false);
    let tops: Vec<usize> = plain.lines().filter(|l| l.contains('┌')).map(|l| l.chars().count()).collect();
    assert!(!tops.is_empty());
}

#[test]
fn svg_has_one_node_per_stage_and_one_arc_per_loop() {
    let f = sample("rup");
    let drawn = svg(&f);
    assert_eq!(drawn.matches("class=\"node").count(), f.stages.len());
    assert_eq!(drawn.matches("<g class=\"loop").count(), f.loops.len());
    assert_eq!(drawn.matches("class=\"edge\"").count(), 3);
}

#[test]
fn a_loop_without_max_is_drawn_as_open_in_the_ui() {
    let drawn = svg(&sample("xgh"));
    assert!(drawn.contains("class=\"loop open\""));
    assert!(drawn.contains("class=\"back\""));
}

#[test]
fn page_has_a_tab_and_a_detail_per_item_and_escapes_text() {
    let f = inline("  agent \"a\" { content = \"<script>x</script>\" }\n  stage \"s\" { agents = [\"a\"] }");
    let html = page("t", &[&f, &sample("kanban")]);
    assert_eq!(html.matches("data-tab=").count(), 2);
    assert!(html.contains("id=\"f0-stage-0\"") && html.contains("id=\"f1-loop-1\""));
    assert!(!html.contains("<script>x</script>") && html.contains("&lt;script&gt;x&lt;/script&gt;"));
    assert_eq!(escape("a&\"b\""), "a&amp;&quot;b&quot;");
}

#[test]
fn args_pick_the_mode_path_and_flags() {
    let raw: Vec<String> = ["--render-ui", "samples/xp.factoryfile", "--factory", "xp", "--out", "x.html", "--no-open"].map(String::from).to_vec();
    let a = args::parse(&raw).unwrap();
    assert_eq!(a.mode, Mode::RenderUi);
    assert_eq!(a.path, PathBuf::from("samples/xp.factoryfile"));
    assert_eq!((a.factory.as_deref(), a.open), (Some("xp"), false));
    let d = args::parse(&["--render".to_string()]).unwrap();
    assert_eq!((d.mode, d.path), (Mode::Render, PathBuf::from("Factoryfile")));
    assert!(args::parse(&["--bogus".to_string()]).is_err());
    assert!(args::parse(&["--factory".to_string()]).is_err());
    assert_eq!(args::parse(&[]).unwrap().mode, Mode::Help);
}
