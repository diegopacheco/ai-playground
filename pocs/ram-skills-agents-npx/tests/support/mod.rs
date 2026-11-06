use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn temp_dir(label: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("ram-test-{label}-{}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn write(root: &Path, rel: &str, body: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

pub fn fixture_repo() -> PathBuf {
    let root = temp_dir("repo");
    write(&root, "bash-skill/SKILL.md", "---\nname: bash-skill\ndescription: Bash rules\n---\nbody");
    write(&root, "bash-skill/install.sh", "echo hi");
    write(&root, "bash-skill/node_modules/junk/index.js", "x");
    write(&root, "patterns/01-x/skill/SKILL.md", "---\nname: sql-migration\ndescription: \"Plan migrations\"\n---\n");
    write(&root, "patterns/01-x/skill/references/pg.md", "pg");
    write(&root, "fact-checker/SKILL.md", "---\nname: fact-checker\ndescription: Checks facts\n---\n");
    write(&root, "fact-checker/command.md", "# /fact-check\n\nUsage: `/fact-check <claim>`\n");
    write(&root, "commands/do.md", "# /do\n\nUsage: `/do <prompt>`\n");
    write(&root, "commands/README.md", "# Commands\n");
    write(&root, "team/agents/reviewer.md", "---\nname: code-reviewer\ndescription: Reviews code\n---\nYou review.");
    write(&root, "node_modules/evil/SKILL.md", "---\nname: evil\n---\n");
    write(
        &root,
        ".claude-plugin/marketplace.json",
        r#"{"name":"diego-market","plugins":[{"name":"night-shift","source":"./plugins/night-shift","description":"Night agents plugin"}]}"#,
    );
    root
}
