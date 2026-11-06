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

fn routes() -> Factory {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spec/fixtures/valid/routes.factoryfile");
    parse_file(path).unwrap().factories.remove(0)
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
    let back: Vec<(String, String)> = g.backward.iter().map(|e| (f.stages[e.from].name.clone(), f.stages[e.to].name.clone())).collect();
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

fn node_x(drawn: &str, stage: usize) -> usize {
    let marker = format!("data-target=\"stage-{stage}\" tabindex=\"0\"><rect x=\"");
    let start = drawn.find(&marker).unwrap() + marker.len();
    drawn[start..].split('"').next().unwrap().parse().unwrap()
}

#[test]
fn when_every_stage_sets_at_the_rows_are_the_layers_and_columns_order_them() {
    let f = routes();
    let flat: Vec<Vec<String>> = layout(&f).layers.iter().map(|l| names(&f, l)).collect();
    assert_eq!(flat, [vec!["implement", "test", "approve", "split"], vec!["done", "ship", "build"]]);
    let drawn = svg(&f);
    let x = |name: &str| node_x(&drawn, f.stages.iter().position(|s| s.name == name).unwrap());
    assert_eq!(x("test"), x("done"), "both sit in column 1");
    assert_eq!(x("split"), x("build"), "both sit in column 3");
    assert!(x("implement") < x("test") && x("test") < x("approve"));
}

#[test]
fn one_stage_without_at_falls_back_to_layers_over_next_and_routes() {
    let f = inline(
        "  stage \"a\" {\n    run = \"test\"\n    at = [\"5\", \"5\"]\n    on \"passed\" { to = \"b\" }\n  }\n  stage \"b\" { run = \"ship\" next = [\"c\"] }\n  stage \"c\" { run = \"finish\" }",
    );
    let flat: Vec<Vec<String>> = layout(&f).layers.iter().map(|l| names(&f, l)).collect();
    assert_eq!(flat, [["a"], ["b"], ["c"]], "routes push their target to the next layer");
}

#[test]
fn a_route_back_is_bounded_only_by_its_own_cycle_or_exhausted_like_the_linter() {
    let g = layout(&routes());
    let f = routes();
    let back: Vec<(String, String, bool)> =
        g.backward.iter().map(|e| (f.stages[e.from].name.clone(), f.stages[e.to].name.clone(), e.bounded)).collect();
    assert_eq!(
        back,
        [("approve".to_string(), "implement".to_string(), true), ("test".to_string(), "implement".to_string(), false)]
    );
    let open = inline("  stage \"a\" { run = \"test\" next = [\"b\"] }\n  stage \"b\" {\n    run = \"test\"\n    on \"failed\" { to = \"a\" }\n  }");
    assert!(!layout(&open).backward[0].bounded);
}

#[test]
fn terminal_render_shows_routes_fallbacks_and_commands() {
    let text = render(&routes(), false);
    assert!(text.contains("└▶ approve  on passed · pass"), "{text}");
    assert!(text.contains("↩ implement  on rejected\n") || text.contains("↩ implement  on rejected "), "{text}");
    assert!(text.contains("↩ implement  on failed · fail (unbounded)"), "{text}");
    assert!(text.contains("┄▶ approve  fallback of next · first fallback"), "{text}");
    assert!(text.contains("┄▶ done  fallback of next · last fallback"), "{text}");
    assert!(text.contains("run test") && text.contains("run script scripts/ship.sh"), "{text}");
    assert!(text.contains("■ end succeeded"), "{text}");
    assert!(!text.contains("no agents"), "command stages show their run in place of agents");
}

#[test]
fn terminal_render_flags_a_route_back_that_nothing_bounds() {
    let f = inline("  stage \"a\" { run = \"test\" next = [\"b\"] }\n  stage \"b\" {\n    run = \"test\"\n    on \"failed\" { to = \"a\" }\n  }");
    let text = render(&f, false);
    assert!(text.contains("↩ a  on failed (unbounded)"), "{text}");
    assert!(text.contains("↩ on failed b ──▶ a  no max, it may never end"), "{text}");
}

fn edge_with_label<'a>(drawn: &'a str, label: &str) -> &'a str {
    let end = drawn.find(&format!(">{label}</text>")).unwrap();
    &drawn[drawn[..end].rfind("<g class=").unwrap()..end]
}

fn path_points(edge: &str) -> Vec<(i64, i64)> {
    let start = edge.find("d=\"").unwrap() + 3;
    let d = &edge[start..start + edge[start..].find('"').unwrap()];
    let numbers: Vec<i64> = d.split(|c: char| !c.is_ascii_digit() && c != '-').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    numbers.chunks(2).map(|p| (p[0], p[1])).collect()
}

fn labels(drawn: &str) -> Vec<(i64, i64, i64)> {
    drawn
        .split("<text class=\"edge-label\" x=\"")
        .skip(1)
        .map(|rest| {
            let x: i64 = rest.split('"').next().unwrap().parse().unwrap();
            let y: i64 = rest.split("y=\"").nth(1).unwrap().split('"').next().unwrap().parse().unwrap();
            let text = rest.split('>').nth(1).unwrap().split('<').next().unwrap();
            (x, y, text.chars().count() as i64 * 6 + 10)
        })
        .collect()
}

#[test]
fn with_at_an_edge_between_neighbouring_cells_is_a_short_straight_line_in_either_direction() {
    let drawn = svg(&sample("gated-release"));
    for label in ["next · retest", "failed · red", "approved · yes", "passed · green", "still red"] {
        let points = path_points(edge_with_label(&drawn, label));
        assert_eq!(points.len(), 2, "{label} must be one straight segment: {points:?}");
        assert!(points[0].0 == points[1].0 || points[0].1 == points[1].1, "{label}: {points:?}");
    }
    let up = path_points(edge_with_label(&drawn, "next · retest"));
    let down = path_points(edge_with_label(&drawn, "failed · red"));
    assert_ne!(up[0].0, down[0].0, "the two directions between test and fix stay apart");
}

#[test]
fn with_at_an_edge_that_goes_around_runs_between_rows_not_along_a_row_of_nodes() {
    let f = sample("gated-release");
    let drawn = svg(&f);
    let row_bottom = path_points(edge_with_label(&drawn, "passed · green"))[0].1 + 42;
    let next_row_top = path_points(edge_with_label(&drawn, "approved · yes"))[1].1;
    for label in ["next · next cycle", "rejected · no", "failed · release failed"] {
        let points = path_points(edge_with_label(&drawn, label));
        let ys: Vec<i64> = points.windows(2).filter(|s| s[0].1 == s[1].1).map(|s| s[0].1).collect();
        assert!(!ys.is_empty() && ys.iter().all(|y| *y > row_bottom && *y < next_row_top), "{label}: {points:?}");
    }
}

#[test]
fn every_edge_label_is_readable_without_touching_another_label() {
    for f in [sample("gated-release"), routes()] {
        let found = labels(&svg(&f));
        assert!(found.len() >= 7, "{found:?}");
        for (i, a) in found.iter().enumerate() {
            for b in &found[i + 1..] {
                let apart = (a.0 - b.0).abs() * 2 >= a.2 + b.2 || (a.1 - b.1).abs() >= 16;
                assert!(apart, "{}: labels at {a:?} and {b:?} overlap", f.name);
            }
        }
    }
}

#[test]
fn the_graph_scales_down_to_the_panel_instead_of_being_cut() {
    let html = page("t", &[&sample("gated-release")]);
    assert!(html.contains(".graph{display:block;margin:0 auto;max-width:100%;height:auto}"));
    assert!(svg(&sample("gated-release")).contains("viewBox=\"0 0 "));
}

#[test]
fn svg_draws_routes_with_labels_fallbacks_dashed_and_commands_as_run() {
    let drawn = svg(&routes());
    assert_eq!(drawn.matches("class=\"fallback\"").count(), 2);
    assert!(drawn.contains(">first fallback</text>") && drawn.contains(">last fallback</text>"));
    assert!(drawn.contains(">passed · pass</text>") && drawn.contains(">approved</text>"));
    assert_eq!(drawn.matches("class=\"edge route\"").count(), 3);
    assert_eq!(drawn.matches("class=\"back bounded\"").count(), 1, "approve rejected starts a new cycle");
    assert_eq!(drawn.matches("class=\"back\"").count(), 1, "test failed has no cycle and no fallback");
    assert!(drawn.contains(">run test</text>") && drawn.contains(">command</text>"));
    assert!(page("t", &[&routes()]).contains(".fallback{fill:none;stroke:#94a3b8;stroke-width:1.4;stroke-dasharray:4 4}"));
}
