use crate::github::types::FetchedCommit;

pub fn build_batch_prompt(commits: &[FetchedCommit]) -> String {
    let mut prompt = String::from(
        "You are a ruthless, no-bullshit senior staff engineer reviewing git commits. \
         Your job is to classify each commit's REAL depth and value. Be HARSH and HONEST. \
         Most commits in the world are SHALLOW - do not sugarcoat.\n\n\
         CLASSIFICATION RULES (follow strictly):\n\
         SHALLOW (score 1-3): This is the DEFAULT. Assign SHALLOW unless there is clear evidence otherwise. Includes:\n\
         - ANY commit that only adds files without meaningful logic (boilerplate, scaffolding, config, generated code)\n\
         - Version bumps, dependency updates, lock file changes\n\
         - Typo fixes, README updates, documentation-only changes\n\
         - Renaming, reformatting, moving files around\n\
         - Adding empty projects, starter templates, hello-world code\n\
         - Copy-paste code with minimal modification\n\
         - Adding shell scripts that just start/stop services\n\
         - Auto-generated code, Cargo.lock, package-lock.json, go.sum changes\n\
         - Commits with vague messages like 'added X', 'update Y', 'fix Z' with trivial diffs\n\
         - POC/playground commits that are just experimenting\n\n\
         DECENT (score 4-6): Assign ONLY when the commit shows real thought:\n\
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
         IMPORTANT: Look at the DIFF, not just the message. A commit message saying 'added feature X' \
         means NOTHING if the diff is just boilerplate. Judge by the ACTUAL CODE CHANGES.\n\n\
         If multiple commits are from the same day and same repo, consider them as a group - \
         repeated trivial commits on the same day should lower the score, not inflate it.\n\n\
         Respond ONLY with valid JSON in this exact format:\n\
         {\"results\": [{\"index\": 1, \"classification\": \"SHALLOW\", \"summary\": \"one line summary\", \"score\": 2}, ...]}\n\n\
         Here are the commits to analyze:\n\n",
    );

    for (i, commit) in commits.iter().enumerate() {
        prompt.push_str(&format!(
            "[{}] Repo: {}\nSHA: {}\nDate: {}\nMessage: {}\nDiff:\n{}\n\n",
            i + 1,
            commit.repo_name,
            commit.sha,
            commit.date,
            commit.message,
            if commit.diff.is_empty() { "(no diff available)" } else { &commit.diff }
        ));
    }

    prompt
}
