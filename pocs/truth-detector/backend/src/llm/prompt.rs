use crate::github::types::FetchedCommit;

pub fn build_batch_prompt(commits: &[FetchedCommit]) -> String {
    let mut prompt = String::from(
        "Analyze the following git commits and classify each one. For each commit, determine:\n\
         - classification: DEEP (substantial architectural change, complex logic, significant feature, non-trivial bug fix, score 7-10), DECENT (meaningful contribution, moderate complexity, clear purpose, score 4-6), or SHALLOW (config tweak, version bump, typo fix, auto-generated, trivial rename, score 1-3)\n\
         - summary: a one-line summary of what the commit does\n\
         - score: 1-10 based on the classification criteria above\n\n\
         Respond ONLY with valid JSON in this exact format:\n\
         {\"results\": [{\"index\": 1, \"classification\": \"DEEP\", \"summary\": \"one line summary\", \"score\": 8}, ...]}\n\n\
         Here are the commits:\n\n",
    );

    for (i, commit) in commits.iter().enumerate() {
        prompt.push_str(&format!(
            "[{}] Repo: {}\nSHA: {}\nMessage: {}\nDiff:\n{}\n\n",
            i + 1,
            commit.repo_name,
            commit.sha,
            commit.message,
            commit.diff
        ));
    }

    prompt
}
