package com.github.diegopacheco.devadminconsole.engine;

import java.util.List;
import java.util.Map;

public record QueryResult(List<String> columns, List<Map<String, Object>> rows, long elapsedMs, int pageNumber,
                          String nextCursor, boolean hasMore, Long totalRows) {
    public static QueryResult of(List<String> columns, List<Map<String, Object>> rows, int pageNumber,
                                 String nextCursor, boolean hasMore) {
        return new QueryResult(columns, rows, 0, pageNumber, nextCursor, hasMore, null);
    }

    public QueryResult withElapsed(long elapsedMs) {
        return new QueryResult(columns, rows, elapsedMs, pageNumber, nextCursor, hasMore, totalRows);
    }
}
