package com.github.diegopacheco.adminconsole.federation;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.stereotype.Component;

@Component
public class FederatedQueryParser {
    private static final Pattern SHAPE = Pattern.compile(
            "^\\s*SELECT\\s+(?<projection>.+?)\\s+"
                    + "FROM\\s+(?<leftSource>[\\w.\\-/:*]+)\\s+(?:AS\\s+)?(?<leftAlias>\\w+)\\s*"
                    + "(?:WHERE\\s+(?<leftWhere>.+?)\\s+)?"
                    + "(?<joinType>LEFT\\s+JOIN|INNER\\s+JOIN|JOIN)\\s+(?<rightSource>[\\w.\\-/:*]+)\\s+(?:AS\\s+)?(?<rightAlias>\\w+)\\s*"
                    + "ON\\s+(?<leftKey>\\w+\\.\\w+)\\s*=\\s*(?<rightKey>\\w+\\.\\w+)\\s*"
                    + "(?:WHERE\\s+(?<rightWhere>.+?)\\s*)?"
                    + "(?:LIMIT\\s+(?<limit>\\d+)\\s*)?$",
            Pattern.CASE_INSENSITIVE | Pattern.DOTALL);

    private static final int DEFAULT_LIMIT = 100;
    private static final int MAX_LIMIT = 1000;

    public FederatedQuery parse(String statement) {
        if (statement == null || statement.isBlank()) {
            throw new IllegalArgumentException("write a federated SELECT");
        }
        String normalized = statement.trim().replaceAll(";\\s*$", "").replaceAll("\\s+", " ");
        Matcher matcher = SHAPE.matcher(normalized);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("""
                    expected: SELECT cols FROM <connection>.<source> a JOIN <connection>.<source> b ON a.x = b.y [LIMIT n]
                    only one equality join between two sources is supported""");
        }

        List<String> projection = new ArrayList<>();
        for (String column : matcher.group("projection").split(",")) {
            projection.add(column.trim());
        }
        if (projection.isEmpty()) {
            throw new IllegalArgumentException("select at least one column");
        }

        String leftAlias = matcher.group("leftAlias");
        String rightAlias = matcher.group("rightAlias");
        if (leftAlias.equalsIgnoreCase(rightAlias)) {
            throw new IllegalArgumentException("the two sides need different aliases");
        }

        FederatedQuery.Side left = side(matcher.group("leftSource"), leftAlias, matcher.group("leftWhere"));
        FederatedQuery.Side right = side(matcher.group("rightSource"), rightAlias, matcher.group("rightWhere"));

        String rawLeftKey = matcher.group("leftKey");
        String rawRightKey = matcher.group("rightKey");
        String leftKey = keyFor(leftAlias, rawLeftKey, rawRightKey);
        String rightKey = keyFor(rightAlias, rawLeftKey, rawRightKey);

        int limit = matcher.group("limit") == null
                ? DEFAULT_LIMIT
                : Math.min(Integer.parseInt(matcher.group("limit")), MAX_LIMIT);

        boolean leftJoin = matcher.group("joinType").toUpperCase().startsWith("LEFT");
        return new FederatedQuery(projection, left, right, leftKey, rightKey, leftJoin, limit);
    }

    private FederatedQuery.Side side(String qualified, String alias, String where) {
        int separator = qualified.indexOf('.');
        if (separator <= 0 || separator == qualified.length() - 1) {
            throw new IllegalArgumentException(
                    "each side must be <connection>.<source>, for example demo-postgres.customers — got: " + qualified);
        }
        return new FederatedQuery.Side(alias, qualified.substring(0, separator), qualified.substring(separator + 1),
                where == null ? null : where.trim());
    }

    private String keyFor(String alias, String first, String second) {
        for (String candidate : List.of(first, second)) {
            String[] parts = candidate.split("\\.", 2);
            if (parts[0].equalsIgnoreCase(alias)) {
                return parts[1];
            }
        }
        throw new IllegalArgumentException("the ON clause must reference both aliases, for example ON "
                + alias + ".id = other.id");
    }
}
