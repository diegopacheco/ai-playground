package com.github.controlpanel.issues;

import com.github.controlpanel.common.Encoding;
import com.github.controlpanel.common.Label;
import com.github.controlpanel.common.Times;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class IssueService {

    private final JdbcClient jdbc;

    public IssueService(JdbcClient jdbc) {
        this.jdbc = jdbc;
    }

    public record IssueListItem(long id, String repo, int number, String title, String author, String state,
                                int commentsCount, List<Label> labels, String updatedAt, String url) {
    }

    public record IssueDetail(long id, String repo, int number, String title, String author, String state, String body,
                              int commentsCount, List<String> assignees, List<Label> labels,
                              String createdAt, String updatedAt, String closedAt, String url) {
    }

    public List<IssueListItem> list() {
        return jdbc.sql("""
                        SELECT i.id, r.full_name AS repo, i.number, i.title, i.author, i.state,
                               i.comments_count, i.labels, i.updated_at, i.url
                        FROM issue i
                        JOIN repository r ON r.id = i.repository_id
                        ORDER BY i.updated_at DESC
                        """)
                .query((rs, n) -> new IssueListItem(
                        rs.getLong("id"),
                        rs.getString("repo"),
                        rs.getInt("number"),
                        rs.getString("title"),
                        rs.getString("author"),
                        rs.getString("state"),
                        rs.getInt("comments_count"),
                        Encoding.decodeLabels(rs.getString("labels")),
                        Times.iso(rs.getTimestamp("updated_at")),
                        rs.getString("url")))
                .list();
    }

    public IssueDetail get(long id) {
        return jdbc.sql("""
                        SELECT i.id, r.full_name AS repo, i.number, i.title, i.author, i.state, i.body,
                               i.comments_count, i.assignees, i.labels, i.created_at, i.updated_at, i.closed_at, i.url
                        FROM issue i
                        JOIN repository r ON r.id = i.repository_id
                        WHERE i.id = ?
                        """)
                .param(id)
                .query((rs, n) -> new IssueDetail(
                        rs.getLong("id"),
                        rs.getString("repo"),
                        rs.getInt("number"),
                        rs.getString("title"),
                        rs.getString("author"),
                        rs.getString("state"),
                        rs.getString("body"),
                        rs.getInt("comments_count"),
                        Encoding.decodeList(rs.getString("assignees")),
                        Encoding.decodeLabels(rs.getString("labels")),
                        Times.iso(rs.getTimestamp("created_at")),
                        Times.iso(rs.getTimestamp("updated_at")),
                        Times.iso(rs.getTimestamp("closed_at")),
                        rs.getString("url")))
                .optional()
                .orElse(null);
    }
}
