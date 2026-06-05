package com.github.controlpanel.dashboard;

import com.github.controlpanel.common.Times;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DashboardService {

    private final JdbcClient jdbc;

    public DashboardService(JdbcClient jdbc) {
        this.jdbc = jdbc;
    }

    public record IssueCounts(long open, long closed, long total) {
    }

    public record PrCounts(long open, long closed, long merged, long draft, long pendingReview, long total) {
    }

    public record Contributor(String login, long commits, long prsOpened, long issuesOpened, long reviews, long total) {
    }

    public record RepoSummary(String fullName, long openIssues, long openPrs, String lastSyncedAt) {
    }

    public record Dashboard(int repoCount, IssueCounts issues, PrCounts prs,
                            List<Contributor> contributors, List<RepoSummary> repos) {
    }

    public Dashboard get() {
        int repoCount = count("SELECT COUNT(*) FROM repository").intValue();

        long openIssues = count("SELECT COUNT(*) FROM issue WHERE state = 'OPEN'");
        long totalIssues = count("SELECT COUNT(*) FROM issue");
        IssueCounts issues = new IssueCounts(openIssues, totalIssues - openIssues, totalIssues);

        long openPrs = count("SELECT COUNT(*) FROM pull_request WHERE state = 'OPEN'");
        long closedPrs = count("SELECT COUNT(*) FROM pull_request WHERE state = 'CLOSED'");
        long mergedPrs = count("SELECT COUNT(*) FROM pull_request WHERE state = 'MERGED'");
        long draftPrs = count("SELECT COUNT(*) FROM pull_request WHERE state = 'OPEN' AND draft = TRUE");
        long pendingPrs = count("SELECT COUNT(*) FROM pull_request WHERE state = 'OPEN' AND review_requests_count > 0");
        PrCounts prs = new PrCounts(openPrs, closedPrs, mergedPrs, draftPrs, pendingPrs, openPrs + closedPrs + mergedPrs);

        List<Contributor> contributors = jdbc.sql("""
                        SELECT login,
                               SUM(commits) AS commits,
                               SUM(prs_opened) AS prs_opened,
                               SUM(issues_opened) AS issues_opened,
                               SUM(reviews) AS reviews,
                               SUM(commits + prs_opened + issues_opened + reviews) AS total
                        FROM contribution
                        GROUP BY login
                        ORDER BY total DESC
                        LIMIT 25
                        """)
                .query((rs, n) -> new Contributor(
                        rs.getString("login"),
                        rs.getLong("commits"),
                        rs.getLong("prs_opened"),
                        rs.getLong("issues_opened"),
                        rs.getLong("reviews"),
                        rs.getLong("total")))
                .list();

        List<RepoSummary> repos = jdbc.sql("""
                        SELECT r.full_name AS full_name,
                               (SELECT COUNT(*) FROM issue i WHERE i.repository_id = r.id AND i.state = 'OPEN') AS open_issues,
                               (SELECT COUNT(*) FROM pull_request p WHERE p.repository_id = r.id AND p.state = 'OPEN') AS open_prs,
                               r.last_synced_at AS last_synced_at
                        FROM repository r
                        ORDER BY r.full_name
                        """)
                .query((rs, n) -> new RepoSummary(
                        rs.getString("full_name"),
                        rs.getLong("open_issues"),
                        rs.getLong("open_prs"),
                        Times.iso(rs.getTimestamp("last_synced_at"))))
                .list();

        return new Dashboard(repoCount, issues, prs, contributors, repos);
    }

    private Long count(String sql) {
        Long value = jdbc.sql(sql).query(Long.class).single();
        return value == null ? 0L : value;
    }
}
