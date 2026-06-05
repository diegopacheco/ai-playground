package com.github.controlpanel.sync;

import com.github.controlpanel.common.Encoding;
import com.github.controlpanel.common.Times;
import com.github.controlpanel.github.RepoData;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashMap;
import java.util.Map;

@Component
public class RepoWriter {

    private final JdbcClient jdbc;

    public RepoWriter(JdbcClient jdbc) {
        this.jdbc = jdbc;
    }

    @Transactional
    public int replace(Long repoId, RepoData.Repo data) {
        jdbc.sql("DELETE FROM pull_request WHERE repository_id = ?").param(repoId).update();
        jdbc.sql("DELETE FROM issue WHERE repository_id = ?").param(repoId).update();
        jdbc.sql("DELETE FROM contribution WHERE repository_id = ?").param(repoId).update();

        Map<String, int[]> contributions = new HashMap<>();
        for (RepoData.ContributorData contributor : data.contributors()) {
            contributions.computeIfAbsent(contributor.login(), k -> new int[4])[0] += contributor.contributions();
        }

        for (RepoData.PullData pr : data.pulls()) {
            insertPr(repoId, pr);
            bump(contributions, pr.author(), 1);
        }

        for (RepoData.IssueData issue : data.issues()) {
            insertIssue(repoId, issue);
            bump(contributions, issue.author(), 2);
        }

        contributions.forEach((login, counts) -> jdbc.sql(
                        "INSERT INTO contribution (repository_id, login, commits, prs_opened, issues_opened, reviews) VALUES (?, ?, ?, ?, ?, ?)")
                .params(repoId, login, counts[0], counts[1], counts[2], counts[3]).update());

        jdbc.sql("UPDATE repository SET last_synced_at = ? WHERE id = ?").params(Times.now(), repoId).update();
        return data.pulls().size();
    }

    private void insertPr(Long repoId, RepoData.PullData pr) {
        jdbc.sql("""
                        INSERT INTO pull_request
                        (repository_id, number, title, author, state, draft, ci_status, mergeable, review_decision,
                         review_requests_count, labels, reviewers, created_at, updated_at, closed_at, merged_at, url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """)
                .params(repoId, pr.number(), pr.title(), pr.author(), pr.state(), pr.draft(), null, null, null,
                        pr.requestedReviewers(), Encoding.encodeLabels(pr.labels()), null,
                        Times.parse(pr.createdAt()), Times.parse(pr.updatedAt()), Times.parse(pr.closedAt()),
                        Times.parse(pr.mergedAt()), pr.url())
                .update();
    }

    private void insertIssue(Long repoId, RepoData.IssueData issue) {
        jdbc.sql("""
                        INSERT INTO issue
                        (repository_id, number, title, author, state, body, comments_count, assignees_count,
                         assignees, labels, created_at, updated_at, closed_at, url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """)
                .params(repoId, issue.number(), issue.title(), issue.author(), issue.state(), issue.body(),
                        issue.comments(), issue.assigneesCount(),
                        Encoding.encodeList(issue.assignees()), Encoding.encodeLabels(issue.labels()),
                        Times.parse(issue.createdAt()), Times.parse(issue.updatedAt()), Times.parse(issue.closedAt()),
                        issue.url())
                .update();
    }

    private static void bump(Map<String, int[]> contributions, String login, int index) {
        if (login == null) {
            return;
        }
        contributions.computeIfAbsent(login, k -> new int[4])[index]++;
    }
}
