package com.github.diegopacheco.adminconsole.federation;

import java.util.List;

public record FederatedQuery(List<String> projection, List<Side> sides, List<Join> joins, int limit) {

    public record Side(String alias, String connectionName, String source, String where) {}

    public record Join(String leftAlias, String leftKey, String rightAlias, String rightKey, boolean leftJoin) {}

    public Side sideOf(String alias) {
        return sides.stream()
                .filter(side -> side.alias().equalsIgnoreCase(alias))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown alias: " + alias));
    }
}
