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

        long joins = java.util.regex.Pattern.compile("\\bJOIN\\b", Pattern.CASE_INSENSITIVE)
                .matcher(normalized).results().count();
        if (joins > 1) {
            throw new IllegalArgumentException("found " + joins + " JOIN clauses — this console joins exactly two "
                    + "sources. Join two of them here, then use the result to narrow a third query.");
        }
        if (joins == 0) {
            throw new IllegalArgumentException("no JOIN found — a cross-engine query needs two sources, for example: "
                    + "SELECT a.id, b.name FROM demo-mysql.invoices a JOIN demo-elasticsearch.products b ON a.id = b._id");
        }

        Matcher matcher = SHAPE.matcher(normalized);
        if (!matcher.matches()) {
            throw new IllegalArgumentException(diagnose(normalized));
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

    String diagnose(String normalized) {
        Matcher limit = Pattern.compile("\\bLIMIT\\s+(\\S+)", Pattern.CASE_INSENSITIVE).matcher(normalized);
        if (limit.find() && !limit.group(1).chars().allMatch(Character::isDigit)) {
            return "LIMIT must be a whole number — got \"" + limit.group(1) + "\"";
        }
        if (!Pattern.compile("\\bSELECT\\b", Pattern.CASE_INSENSITIVE).matcher(normalized).find()) {
            return "the statement must start with SELECT";
        }
        if (!Pattern.compile("\\bFROM\\b", Pattern.CASE_INSENSITIVE).matcher(normalized).find()) {
            return "no FROM clause — name the first source as <connection>.<source>";
        }
        if (!Pattern.compile("\\bON\\b", Pattern.CASE_INSENSITIVE).matcher(normalized).find()) {
            return "the JOIN has no ON clause — add ON <alias>.<column> = <alias>.<column>";
        }
        Matcher on = Pattern.compile("\\bON\\s+(.+?)(?:\\s+LIMIT\\b|$)", Pattern.CASE_INSENSITIVE).matcher(normalized);
        if (on.find()) {
            String condition = on.group(1).trim();
            if (condition.toUpperCase().contains(" AND ") || condition.toUpperCase().contains(" OR ")) {
                return "only one equality is supported in ON — got: " + condition;
            }
            if (!condition.matches("\\w+\\.\\w+\\s*=\\s*\\w+\\.\\w+")) {
                return "ON must compare two alias-qualified columns, for example ON a.id = b.key — got: " + condition;
            }
        }
        if (!Pattern.compile("\\bFROM\\s+[\\w.\\-/:*]+\\s+(?:AS\\s+)?\\w+", Pattern.CASE_INSENSITIVE)
                .matcher(normalized).find()) {
            return "each source needs an alias, for example FROM demo-mysql.invoices a";
        }
        return """
                expected: SELECT cols FROM <connection>.<source> a JOIN <connection>.<source> b ON a.x = b.y [LIMIT n]
                only one equality join between two sources is supported""";
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
