use ram::text::{width, wrap};

#[test]
fn long_lines_wrap_on_words_within_the_terminal_width() {
    let lines = wrap("the quick brown fox jumps over the lazy dog", 15);
    assert_eq!(lines, vec!["the quick brown", "fox jumps over", "the lazy dog"]);
    assert!(lines.iter().all(|l| width(l) <= 15));
}

#[test]
fn bullets_keep_their_indent_when_wrapped() {
    let lines = wrap("  * one two three four five", 14);
    assert_eq!(lines, vec!["  * one two", "  three four", "  five"]);
}

#[test]
fn words_longer_than_the_width_are_split_instead_of_overflowing() {
    let lines = wrap("abcdefghijklmnop", 8);
    assert_eq!(lines, vec!["abcdefgh", "ijklmnop"]);
}

#[test]
fn emoji_count_as_two_columns() {
    assert_eq!(width("🏛️ ab"), 5);
}
