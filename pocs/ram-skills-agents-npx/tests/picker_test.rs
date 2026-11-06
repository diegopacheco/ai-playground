use ram::picker::{Entry, Picker};

fn entry(name: &str, haystack: &str) -> Entry {
    Entry { name: name.into(), haystack: haystack.into() }
}

fn picker() -> Picker {
    Picker::new(vec![entry("bash-skill", "skill bash-skill"), entry("readme-skill", "skill readme-skill"), entry("do", "command do")])
}

#[test]
fn search_narrows_the_list_case_insensitively() {
    let mut p = picker();
    for c in "SKILL".chars() {
        p.type_char(c);
    }
    assert_eq!(p.visible(), &[0, 1]);
}

#[test]
fn selections_survive_changing_the_search() {
    let mut p = picker();
    p.type_char('d');
    p.type_char('o');
    p.toggle();
    p.clear_query();
    for c in "bash".chars() {
        p.type_char(c);
    }
    p.toggle();
    assert_eq!(p.chosen(), vec![0, 2]);
}

#[test]
fn select_all_only_touches_what_the_search_shows() {
    let mut p = picker();
    for c in "skill".chars() {
        p.type_char(c);
    }
    p.toggle_all_visible();
    assert_eq!(p.chosen(), vec![0, 1]);
    p.toggle_all_visible();
    assert!(p.chosen().is_empty());
}

#[test]
fn new_search_puts_the_cursor_on_the_best_match_so_space_picks_it() {
    let mut p = picker();
    p.down();
    p.down();
    p.type_char('b');
    assert_eq!(p.cursor(), 0);
    p.toggle();
    assert_eq!(p.chosen(), vec![0]);
}

#[test]
fn name_matches_rank_above_description_matches() {
    let mut p = Picker::new(vec![
        entry("policy-guard", "skill policy-guard blocks bash commands"),
        entry("my-bash-tool", "skill my-bash-tool"),
        entry("bash-skill", "skill bash-skill"),
    ]);
    for c in "bash".chars() {
        p.type_char(c);
    }
    assert_eq!(p.visible(), &[2, 1, 0]);
}

#[test]
fn no_match_means_toggle_selects_nothing() {
    let mut p = picker();
    p.type_char('z');
    p.toggle();
    assert!(p.visible().is_empty());
    assert!(p.chosen().is_empty());
}

#[test]
fn emoji_descriptions_are_cut_by_terminal_columns_so_rows_never_wrap() {
    let cut = ram::tui::truncate("🏛️ Architect night agent reviews everything", 12);
    assert_eq!(cut, "🏛 Architec…");
    assert_eq!(ram::tui::truncate("short", 12), "short");
}
