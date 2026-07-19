package com.github.diegopacheco.devadminconsole.engine.jdbc;

import com.github.diegopacheco.devadminconsole.engine.ReadOnlyViolation;
import java.util.Set;
import java.util.regex.Pattern;
import org.springframework.stereotype.Component;

@Component
public class SqlReadOnlyGuard {
    private static final Set<String> ALLOWED_STARTS = Set.of("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "WITH", "TABLE", "VALUES");
    private static final Pattern WRITE_KEYWORDS = Pattern.compile(
            "\\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|MERGE|REPLACE|CALL|DO|COPY|VACUUM|LOCK|SET|COMMIT|ROLLBACK|SAVEPOINT|REINDEX|CLUSTER|ANALYZE|REFRESH|IMPORT|LOAD|HANDLER|INTO\\s+OUTFILE|INTO\\s+DUMPFILE)\\b");

    public void assertReadOnly(String statement) {
        String normalized = normalize(statement);
        if (normalized.isEmpty()) {
            throw new ReadOnlyViolation("statement is empty");
        }
        if (containsStatementSeparator(normalized)) {
            throw new ReadOnlyViolation("multiple statements are not allowed");
        }
        String first = normalized.split("\\s+")[0].toUpperCase();
        if (!ALLOWED_STARTS.contains(first)) {
            throw new ReadOnlyViolation("only read statements are allowed, found: " + first);
        }
        if (first.equals("EXPLAIN") && normalized.toUpperCase().contains("ANALYZE")) {
            throw new ReadOnlyViolation("EXPLAIN ANALYZE executes the statement and is not allowed");
        }
        var matcher = WRITE_KEYWORDS.matcher(stripLiterals(normalized).toUpperCase());
        if (matcher.find()) {
            throw new ReadOnlyViolation("write keyword is not allowed: " + matcher.group(1).trim());
        }
    }

    String stripLiterals(String statement) {
        StringBuilder result = new StringBuilder();
        char quote = 0;
        for (char character : statement.toCharArray()) {
            if (quote != 0) {
                if (character == quote) {
                    quote = 0;
                    result.append(' ');
                }
                continue;
            }
            if (character == '\'' || character == '"' || character == '`') {
                quote = character;
                result.append(' ');
                continue;
            }
            result.append(character);
        }
        return result.toString();
    }

    public String normalize(String statement) {
        if (statement == null) {
            return "";
        }
        StringBuilder result = new StringBuilder();
        boolean inLineComment = false;
        boolean inBlockComment = false;
        char quote = 0;
        for (int index = 0; index < statement.length(); index++) {
            char character = statement.charAt(index);
            char next = index + 1 < statement.length() ? statement.charAt(index + 1) : 0;
            if (inLineComment) {
                if (character == '\n') {
                    inLineComment = false;
                    result.append(' ');
                }
                continue;
            }
            if (inBlockComment) {
                if (character == '*' && next == '/') {
                    inBlockComment = false;
                    index++;
                    result.append(' ');
                }
                continue;
            }
            if (quote != 0) {
                result.append(character);
                if (character == quote) {
                    quote = 0;
                }
                continue;
            }
            if (character == '-' && next == '-') {
                inLineComment = true;
                index++;
                continue;
            }
            if (character == '#') {
                inLineComment = true;
                continue;
            }
            if (character == '/' && next == '*') {
                inBlockComment = true;
                index++;
                continue;
            }
            if (character == '\'' || character == '"' || character == '`') {
                quote = character;
            }
            result.append(character);
        }
        return result.toString().trim().replaceAll(";\\s*$", "").trim();
    }

    private boolean containsStatementSeparator(String normalized) {
        char quote = 0;
        for (char character : normalized.toCharArray()) {
            if (quote != 0) {
                if (character == quote) {
                    quote = 0;
                }
                continue;
            }
            if (character == '\'' || character == '"' || character == '`') {
                quote = character;
                continue;
            }
            if (character == ';') {
                return true;
            }
        }
        return false;
    }
}
