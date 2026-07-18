package com.github.diegopacheco.adminconsole.federation;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.stereotype.Component;

@Component
public class FederatedQueryParser {
    private static final String SOURCE = "[\\w.\\-/:*]+";

    private static final Pattern HEAD = Pattern.compile(
            "^\\s*SELECT\\s+(?<projection>.+?)\\s+FROM\\s+(?<source>" + SOURCE + ")\\s+(?:AS\\s+)?(?<alias>\\w+)\\b",
            Pattern.CASE_INSENSITIVE | Pattern.DOTALL);

    private static final Pattern JOIN_CLAUSE = Pattern.compile(
            "\\b(?<type>LEFT\\s+JOIN|INNER\\s+JOIN|JOIN)\\s+(?<source>" + SOURCE + ")\\s+(?:AS\\s+)?(?<alias>\\w+)\\s+"
                    + "ON\\s+(?<leftKey>\\w+\\.\\w+)\\s*=\\s*(?<rightKey>\\w+\\.\\w+)",
            Pattern.CASE_INSENSITIVE);

    private static final Pattern LIMIT_CLAUSE = Pattern.compile("\\bLIMIT\\s+(\\S+)\\s*$", Pattern.CASE_INSENSITIVE);

    private static final int DEFAULT_LIMIT = 100;
    private static final int MAX_LIMIT = 1000;
    static final int MAX_SIDES = 5;

    public FederatedQuery parse(String statement) {
        if (statement == null || statement.isBlank()) {
            throw new IllegalArgumentException("write a federated SELECT");
        }
        String normalized = statement.trim().replaceAll(";\\s*$", "").replaceAll("\\s+", " ");

        Matcher head = HEAD.matcher(normalized);
        if (!head.find()) {
            throw new IllegalArgumentException(diagnose(normalized));
        }

        List<String> projection = new ArrayList<>();
        for (String column : head.group("projection").split(",")) {
            if (!column.isBlank()) {
                projection.add(column.trim());
            }
        }
        if (projection.isEmpty()) {
            throw new IllegalArgumentException("select at least one column");
        }

        List<FederatedQuery.Side> sides = new ArrayList<>();
        sides.add(side(head.group("source"), head.group("alias"), null));

        List<FederatedQuery.Join> joins = new ArrayList<>();
        Matcher join = JOIN_CLAUSE.matcher(normalized);
        while (join.find()) {
            if (sides.size() >= MAX_SIDES) {
                throw new IllegalArgumentException("at most " + MAX_SIDES + " sources can be joined in one query");
            }
            String alias = join.group("alias");
            sides.add(side(join.group("source"), alias, null));
            joins.add(joinOf(join.group("leftKey"), join.group("rightKey"), alias, sides,
                    join.group("type").toUpperCase().startsWith("LEFT")));
        }

        if (joins.isEmpty()) {
            throw new IllegalArgumentException(diagnose(normalized));
        }

        Set<String> aliases = new LinkedHashSet<>();
        for (FederatedQuery.Side side : sides) {
            if (!aliases.add(side.alias().toLowerCase())) {
                throw new IllegalArgumentException("alias \"" + side.alias() + "\" is used twice — give each source "
                        + "its own alias");
            }
        }

        int limit = DEFAULT_LIMIT;
        Matcher limitClause = LIMIT_CLAUSE.matcher(normalized);
        if (limitClause.find()) {
            String value = limitClause.group(1);
            if (!value.chars().allMatch(Character::isDigit)) {
                throw new IllegalArgumentException("LIMIT must be a whole number — got \"" + value + "\"");
            }
            limit = Math.min(Integer.parseInt(value), MAX_LIMIT);
        }

        return new FederatedQuery(projection, sides, joins, limit);
    }

    private FederatedQuery.Join joinOf(String first, String second, String newAlias,
                                       List<FederatedQuery.Side> sides, boolean leftJoin) {
        String[] a = first.split("\\.", 2);
        String[] b = second.split("\\.", 2);
        boolean firstIsNew = a[0].equalsIgnoreCase(newAlias);
        boolean secondIsNew = b[0].equalsIgnoreCase(newAlias);
        if (firstIsNew == secondIsNew) {
            throw new IllegalArgumentException("the ON clause for \"" + newAlias + "\" must compare it to an earlier "
                    + "source, for example ON " + sides.getFirst().alias() + ".id = " + newAlias + ".key");
        }
        String[] existing = firstIsNew ? b : a;
        String[] added = firstIsNew ? a : b;
        boolean known = sides.stream().anyMatch(side -> side.alias().equalsIgnoreCase(existing[0]));
        if (!known) {
            throw new IllegalArgumentException("unknown alias \"" + existing[0] + "\" in the ON clause");
        }
        return new FederatedQuery.Join(existing[0], existing[1], added[0], added[1], leftJoin);
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
        if (!Pattern.compile("\\bJOIN\\b", Pattern.CASE_INSENSITIVE).matcher(normalized).find()) {
            return "no JOIN found — a cross-engine query needs at least two sources, for example: "
                    + "SELECT a.id, b.name FROM demo-mysql.invoices a JOIN demo-elasticsearch.products b ON a.id = b._id";
        }
        if (!Pattern.compile("\\bON\\b", Pattern.CASE_INSENSITIVE).matcher(normalized).find()) {
            return "a JOIN has no ON clause — add ON <alias>.<column> = <alias>.<column>";
        }
        Matcher on = Pattern.compile("\\bON\\s+(.+?)(?:\\s+(?:LEFT\\s+JOIN|INNER\\s+JOIN|JOIN|LIMIT)\\b|$)",
                Pattern.CASE_INSENSITIVE).matcher(normalized);
        if (on.find()) {
            String condition = on.group(1).trim();
            if (condition.toUpperCase().contains(" AND ") || condition.toUpperCase().contains(" OR ")) {
                return "only one equality is supported per ON — got: " + condition;
            }
            if (!condition.matches("\\w+\\.\\w+\\s*=\\s*\\w+\\.\\w+")) {
                return "ON must compare two alias-qualified columns, for example ON a.id = b.key — got: " + condition;
            }
        }
        if (!Pattern.compile("\\bFROM\\s+" + SOURCE + "\\s+(?:AS\\s+)?\\w+", Pattern.CASE_INSENSITIVE)
                .matcher(normalized).find()) {
            return "each source needs an alias, for example FROM demo-mysql.invoices a";
        }
        return """
                expected: SELECT cols FROM <connection>.<source> a JOIN <connection>.<source> b ON a.x = b.y [LIMIT n]
                further sources can be chained with more JOIN ... ON clauses""";
    }

    private FederatedQuery.Side side(String qualified, String alias, String where) {
        int separator = qualified.indexOf('.');
        if (separator <= 0 || separator == qualified.length() - 1) {
            throw new IllegalArgumentException(
                    "each side must be <connection>.<source>, for example demo-postgres.customers — got: " + qualified);
        }
        return new FederatedQuery.Side(alias, qualified.substring(0, separator), qualified.substring(separator + 1),
                where);
    }
}
