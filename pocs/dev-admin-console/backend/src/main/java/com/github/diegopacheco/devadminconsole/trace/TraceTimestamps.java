package com.github.diegopacheco.devadminconsole.trace;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Component;

@Component
public class TraceTimestamps {
    private static final List<String> HINTS = List.of("_at", "timestamp", "time", "date", "created", "updated");

    public Instant of(Map<String, Object> row) {
        for (Map.Entry<String, Object> entry : row.entrySet()) {
            if (!looksTemporal(entry.getKey()) || entry.getValue() == null) {
                continue;
            }
            Instant parsed = parse(String.valueOf(entry.getValue()));
            if (parsed != null) {
                return parsed;
            }
        }
        return null;
    }

    boolean looksTemporal(String column) {
        String name = column.toLowerCase();
        return HINTS.stream().anyMatch(name::contains);
    }

    Instant parse(String value) {
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            return null;
        }
        if (trimmed.chars().allMatch(Character::isDigit)) {
            try {
                long number = Long.parseLong(trimmed);
                if (trimmed.length() >= 13) {
                    return Instant.ofEpochMilli(number);
                }
                if (trimmed.length() >= 9) {
                    return Instant.ofEpochSecond(number);
                }
                return null;
            } catch (NumberFormatException error) {
                return null;
            }
        }
        try {
            return Instant.parse(trimmed);
        } catch (RuntimeException ignored) {
            // fall through to the looser formats below
        }
        try {
            return OffsetDateTime.parse(trimmed).toInstant();
        } catch (RuntimeException ignored) {
            // fall through
        }
        try {
            return LocalDateTime.parse(trimmed.replace(' ', 'T')).toInstant(ZoneOffset.UTC);
        } catch (RuntimeException ignored) {
            return null;
        }
    }
}
