package com.github.controlpanel.common;

import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

public final class Times {
    private Times() {
    }

    public static LocalDateTime parse(String iso) {
        if (iso == null || iso.isBlank()) {
            return null;
        }
        return LocalDateTime.ofInstant(Instant.parse(iso), ZoneOffset.UTC);
    }

    public static String iso(LocalDateTime value) {
        if (value == null) {
            return null;
        }
        return value.toInstant(ZoneOffset.UTC).toString();
    }

    public static String iso(Timestamp value) {
        if (value == null) {
            return null;
        }
        return value.toInstant().toString();
    }

    public static LocalDateTime now() {
        return LocalDateTime.now(ZoneOffset.UTC);
    }

    public static long daysBetween(LocalDateTime from, LocalDateTime to) {
        if (from == null || to == null) {
            return 0;
        }
        return java.time.Duration.between(from, to).toDays();
    }
}
