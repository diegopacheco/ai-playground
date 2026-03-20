use crate::github::types::FetchedCommit;
use std::collections::BTreeMap;

pub struct DayGroup {
    pub date: String,
    pub commits: Vec<usize>,
}

pub fn group_by_day(commits: &[FetchedCommit]) -> Vec<DayGroup> {
    let mut map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, c) in commits.iter().enumerate() {
        let day = c.date.get(..10).unwrap_or(&c.date).to_string();
        map.entry(day).or_default().push(i);
    }
    map.into_iter()
        .map(|(date, commits)| DayGroup { date, commits })
        .collect()
}

pub fn build_batch_prompt(commits: &[FetchedCommit]) -> String {
    let groups = group_by_day(commits);

    let mut prompt = String::from(
        "You are a ruthless, no-bullshit senior staff engineer reviewing a developer's git activity. \
         Your job is to classify the REAL depth and value of their work on EACH DAY. Be HARSH and HONEST.\n\n\
         You are evaluating DAILY work, not individual commits. Look at ALL commits from the same day \
         together as a body of work. A day with 10 trivial commits is still a SHALLOW day. \
         A day with one brilliant fix is a DEEP day.\n\n\
         CLASSIFICATION RULES (follow strictly):\n\
         SHALLOW (score 1-3): This is the DEFAULT. Assign SHALLOW unless there is clear evidence otherwise. Includes:\n\
         - ANY day where commits only add files without meaningful logic (boilerplate, scaffolding, config, generated code)\n\
         - Version bumps, dependency updates, lock file changes\n\
         - Typo fixes, README updates, documentation-only changes\n\
         - Renaming, reformatting, moving files around\n\
         - Adding empty projects, starter templates, hello-world code\n\
         - Copy-paste code with minimal modification\n\
         - Adding shell scripts that just start/stop services\n\
         - Auto-generated code, Cargo.lock, package-lock.json, go.sum changes\n\
         - Commits with vague messages like 'added X', 'update Y', 'fix Z' with trivial diffs\n\
         - POC/playground commits that are just experimenting\n\
         - Multiple small commits that together still don't amount to meaningful work\n\n\
         DECENT (score 4-6): Assign ONLY when the day's work shows real thought:\n\
         - Implements a working feature with actual business logic\n\
         - Fixes a real bug with a non-obvious solution\n\
         - Adds tests that cover meaningful edge cases\n\
         - Refactors code in a way that genuinely improves design\n\
         - Integration of multiple components with error handling\n\n\
         DEEP (score 7-10): Assign RARELY - this is exceptional work:\n\
         - Novel algorithm or data structure implementation\n\
         - Complex system design: concurrency, distributed systems, performance optimization\n\
         - Security hardening with real threat modeling\n\
         - Major architectural refactor that improves the system fundamentally\n\
         - Production incident fix requiring deep debugging across multiple layers\n\n\
         IMPORTANT: Look at the DIFFS, not just the messages. Commit messages mean NOTHING \
         if the diffs are just boilerplate. Judge by the ACTUAL CODE CHANGES across the whole day.\n\n\
         Respond ONLY with valid JSON in this exact format:\n\
         {\"results\": [{\"day_index\": 1, \"classification\": \"SHALLOW\", \"summary\": \"one line summary of the day's work\", \"score\": 2}, ...]}\n\n",
    );

    prompt.push_str(&format!("There are {} day(s) of work to evaluate:\n\n", groups.len()));

    for (gi, group) in groups.iter().enumerate() {
        prompt.push_str(&format!("=== DAY {} ({}) - {} commit(s) ===\n", gi + 1, group.date, group.commits.len()));
        for &ci in &group.commits {
            let c = &commits[ci];
            prompt.push_str(&format!(
                "  Repo: {}\n  SHA: {}\n  Message: {}\n  Diff:\n  {}\n\n",
                c.repo_name,
                c.sha,
                c.message,
                if c.diff.is_empty() { "(no diff available)" } else { &c.diff }
            ));
        }
        prompt.push('\n');
    }

    prompt
}
