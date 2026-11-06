use ram::frontmatter::{parse, slash_name};

#[test]
fn frontmatter_name_and_description_are_read_and_unquoted() {
    let meta = parse("---\nname: sql-migration\ndescription: \"Plan: write and review\"\n---\nbody");
    assert_eq!(meta.name.as_deref(), Some("sql-migration"));
    assert_eq!(meta.description.as_deref(), Some("Plan: write and review"));
}

#[test]
fn files_without_frontmatter_describe_themselves_with_first_text_line() {
    let meta = parse("# /do\n\nUsage: `/do <prompt>`\n");
    assert_eq!(meta.name, None);
    assert_eq!(meta.description.as_deref(), Some("Usage: /do <prompt>"));
}

#[test]
fn slash_heading_is_the_command_name_users_type() {
    assert_eq!(slash_name("# /fact-check\n\nUsage").as_deref(), Some("fact-check"));
    assert_eq!(slash_name("# Fact checker\n"), None);
}
