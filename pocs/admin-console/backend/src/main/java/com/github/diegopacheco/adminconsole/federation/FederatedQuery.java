package com.github.diegopacheco.adminconsole.federation;

import java.util.List;

public record FederatedQuery(List<String> projection, Side left, Side right, String leftKey, String rightKey,
                             boolean leftJoin, int limit) {

    public record Side(String alias, String connectionName, String source, String where) {}
}
