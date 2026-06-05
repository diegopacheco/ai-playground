package com.github.controlpanel.actioncenter;

import com.github.controlpanel.common.Times;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class ActionCenterService {

    private final JdbcClient jdbc;
    private final int staleDays;

    public ActionCenterService(JdbcClient jdbc, @Value("${app.stale-days}") int staleDays) {
        this.jdbc = jdbc;
        this.staleDays = staleDays;
    }

    public record ActionItem(String repo, String type, int number, String title, String author,
                             String url, long ageDays, String detail) {
    }

    public record ActionCenter(int staleDays,
                               List<ActionItem> prsAwaitingReview,
                               List<ActionItem> prsFailingCi,
                               List<ActionItem> prsOpenTooLong,
                               List<ActionItem> staleIssues,
                               List<ActionItem> unassignedIssues) {
    }

    public ActionCenter get() {
        LocalDateTime now = Times.now();

        List<ActionItem> awaitingReview = jdbc.sql("""
                        SELECT r.full_name AS repo, p.number, p.title, p.author, p.url, p.created_at, p.review_decision
                        FROM pull_request p
                        JOIN repository r ON r.id = p.repository_id
                        WHERE p.state = 'OPEN' AND p.draft = FALSE
                          AND (p.review_decision = 'REVIEW_REQUIRED' OR p.review_requests_count > 0)
                        ORDER BY p.created_at
                        """)
                .query((rs, n) -> new ActionItem(rs.getString("repo"), "PR", rs.getInt("number"), rs.getString("title"),
                        rs.getString("author"), rs.getString("url"),
                        age(rs.getTimestamp("created_at"), now),
                        rs.getString("review_decision") == null ? "review requested" : rs.getString("review_decision")))
                .list();

        List<ActionItem> failingCi = jdbc.sql("""
                        SELECT r.full_name AS repo, p.number, p.title, p.author, p.url, p.created_at, p.ci_status
                        FROM pull_request p
                        JOIN repository r ON r.id = p.repository_id
                        WHERE p.state = 'OPEN' AND p.ci_status IN ('FAILURE', 'ERROR')
                        ORDER BY p.created_at
                        """)
                .query((rs, n) -> new ActionItem(rs.getString("repo"), "PR", rs.getInt("number"), rs.getString("title"),
                        rs.getString("author"), rs.getString("url"),
                        age(rs.getTimestamp("created_at"), now), rs.getString("ci_status")))
                .list();

        List<ActionItem> openTooLong = jdbc.sql("""
                        SELECT r.full_name AS repo, p.number, p.title, p.author, p.url, p.created_at
                        FROM pull_request p
                        JOIN repository r ON r.id = p.repository_id
                        WHERE p.state = 'OPEN'
                        ORDER BY p.created_at
                        """)
                .query((rs, n) -> new ActionItem(rs.getString("repo"), "PR", rs.getInt("number"), rs.getString("title"),
                        rs.getString("author"), rs.getString("url"),
                        age(rs.getTimestamp("created_at"), now), "open"))
                .list()
                .stream().filter(item -> item.ageDays() >= staleDays).toList();

        List<ActionItem> staleIssues = jdbc.sql("""
                        SELECT r.full_name AS repo, i.number, i.title, i.author, i.url, i.updated_at
                        FROM issue i
                        JOIN repository r ON r.id = i.repository_id
                        WHERE i.state = 'OPEN'
                        ORDER BY i.updated_at
                        """)
                .query((rs, n) -> new ActionItem(rs.getString("repo"), "ISSUE", rs.getInt("number"), rs.getString("title"),
                        rs.getString("author"), rs.getString("url"),
                        age(rs.getTimestamp("updated_at"), now), "no activity"))
                .list()
                .stream().filter(item -> item.ageDays() >= staleDays).toList();

        List<ActionItem> unassigned = jdbc.sql("""
                        SELECT r.full_name AS repo, i.number, i.title, i.author, i.url, i.created_at
                        FROM issue i
                        JOIN repository r ON r.id = i.repository_id
                        WHERE i.state = 'OPEN' AND i.assignees_count = 0
                        ORDER BY i.created_at
                        """)
                .query((rs, n) -> new ActionItem(rs.getString("repo"), "ISSUE", rs.getInt("number"), rs.getString("title"),
                        rs.getString("author"), rs.getString("url"),
                        age(rs.getTimestamp("created_at"), now), "unassigned"))
                .list();

        return new ActionCenter(staleDays, awaitingReview, failingCi, openTooLong, staleIssues, unassigned);
    }

    private static long age(java.sql.Timestamp ts, LocalDateTime now) {
        if (ts == null) {
            return 0;
        }
        return Times.daysBetween(ts.toInstant().atZone(java.time.ZoneOffset.UTC).toLocalDateTime(), now);
    }
}
