package com.github.controlpanel.insights;

import com.github.controlpanel.common.Times;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Service;

import java.time.DayOfWeek;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.temporal.TemporalAdjusters;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
public class InsightsService {

    private static final int WEEKS = 12;

    private final JdbcClient jdbc;

    public InsightsService(JdbcClient jdbc) {
        this.jdbc = jdbc;
    }

    public record WeekCount(String week, long count) {
    }

    public record WeekPair(String week, long opened, long closed) {
    }

    public record Bucket(String label, long count) {
    }

    public record Contributor(String login, long total) {
    }

    public record Insights(List<WeekCount> prThroughput, double medianTimeToMergeHours, double avgTimeToMergeHours,
                           List<WeekPair> issueFlow, List<Contributor> contributors, List<Bucket> stalePrBuckets) {
    }

    public Insights get() {
        LocalDateTime now = Times.now();
        List<LocalDate> axis = axis(now);

        List<LocalDateTime> mergedAt = dates("SELECT merged_at FROM pull_request WHERE state = 'MERGED' AND merged_at IS NOT NULL");
        List<WeekCount> throughput = weekly(axis, mergedAt);

        List<LocalDateTime> issuesOpened = dates("SELECT created_at FROM issue");
        List<LocalDateTime> issuesClosed = dates("SELECT closed_at FROM issue WHERE closed_at IS NOT NULL");
        List<WeekPair> issueFlow = pairs(axis, issuesOpened, issuesClosed);

        double[] merge = mergeStats();

        List<Contributor> contributors = jdbc.sql("""
                        SELECT login, SUM(commits + prs_opened + issues_opened + reviews) AS total
                        FROM contribution
                        GROUP BY login
                        ORDER BY total DESC
                        LIMIT 10
                        """)
                .query((rs, n) -> new Contributor(rs.getString("login"), rs.getLong("total")))
                .list();

        List<Bucket> stalePrBuckets = stalePrBuckets(now);

        return new Insights(throughput, merge[0], merge[1], issueFlow, contributors, stalePrBuckets);
    }

    private double[] mergeStats() {
        List<Long> hours = new ArrayList<>(jdbc.sql("SELECT created_at, merged_at FROM pull_request WHERE state = 'MERGED' AND merged_at IS NOT NULL")
                .query((rs, n) -> {
                    LocalDateTime created = rs.getTimestamp("created_at").toLocalDateTime();
                    LocalDateTime merged = rs.getTimestamp("merged_at").toLocalDateTime();
                    return Duration.between(created, merged).toHours();
                })
                .list());
        if (hours.isEmpty()) {
            return new double[]{0, 0};
        }
        hours.sort(Long::compareTo);
        double median = hours.size() % 2 == 1
                ? hours.get(hours.size() / 2)
                : (hours.get(hours.size() / 2 - 1) + hours.get(hours.size() / 2)) / 2.0;
        double avg = hours.stream().mapToLong(Long::longValue).average().orElse(0);
        return new double[]{round(median), round(avg)};
    }

    private List<Bucket> stalePrBuckets(LocalDateTime now) {
        long under1 = 0;
        long under7 = 0;
        long under30 = 0;
        long over30 = 0;
        for (LocalDateTime created : dates("SELECT created_at FROM pull_request WHERE state = 'OPEN'")) {
            long days = Times.daysBetween(created, now);
            if (days < 1) {
                under1++;
            } else if (days < 7) {
                under7++;
            } else if (days < 30) {
                under30++;
            } else {
                over30++;
            }
        }
        return List.of(
                new Bucket("< 1 day", under1),
                new Bucket("1-7 days", under7),
                new Bucket("7-30 days", under30),
                new Bucket("30+ days", over30));
    }

    private List<WeekCount> weekly(List<LocalDate> axis, List<LocalDateTime> dates) {
        Map<LocalDate, Long> counts = emptyAxis(axis);
        for (LocalDateTime date : dates) {
            LocalDate week = monday(date);
            if (counts.containsKey(week)) {
                counts.put(week, counts.get(week) + 1);
            }
        }
        List<WeekCount> result = new ArrayList<>();
        counts.forEach((week, count) -> result.add(new WeekCount(week.toString(), count)));
        return result;
    }

    private List<WeekPair> pairs(List<LocalDate> axis, List<LocalDateTime> opened, List<LocalDateTime> closed) {
        Map<LocalDate, Long> openedCounts = emptyAxis(axis);
        Map<LocalDate, Long> closedCounts = emptyAxis(axis);
        for (LocalDateTime date : opened) {
            LocalDate week = monday(date);
            if (openedCounts.containsKey(week)) {
                openedCounts.put(week, openedCounts.get(week) + 1);
            }
        }
        for (LocalDateTime date : closed) {
            LocalDate week = monday(date);
            if (closedCounts.containsKey(week)) {
                closedCounts.put(week, closedCounts.get(week) + 1);
            }
        }
        List<WeekPair> result = new ArrayList<>();
        for (LocalDate week : axis) {
            result.add(new WeekPair(week.toString(), openedCounts.get(week), closedCounts.get(week)));
        }
        return result;
    }

    private Map<LocalDate, Long> emptyAxis(List<LocalDate> axis) {
        Map<LocalDate, Long> map = new LinkedHashMap<>();
        for (LocalDate week : axis) {
            map.put(week, 0L);
        }
        return map;
    }

    private List<LocalDate> axis(LocalDateTime now) {
        LocalDate currentMonday = now.toLocalDate().with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY));
        List<LocalDate> axis = new ArrayList<>();
        for (int i = WEEKS - 1; i >= 0; i--) {
            axis.add(currentMonday.minusWeeks(i));
        }
        return axis;
    }

    private static LocalDate monday(LocalDateTime date) {
        return date.toLocalDate().with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY));
    }

    private List<LocalDateTime> dates(String sql) {
        return jdbc.sql(sql)
                .query((rs, n) -> rs.getTimestamp(1) == null ? null : rs.getTimestamp(1).toLocalDateTime())
                .list()
                .stream().filter(Objects::nonNull).toList();
    }

    private static double round(double value) {
        return Math.round(value * 10.0) / 10.0;
    }
}
