package com.store.util;

public final class StringUtils {

    private StringUtils() {
    }

    public static String slugify(String value) {
        return value.trim().toLowerCase().replaceAll("[^a-z0-9]+", "-");
    }

    public static String reverse(String value) {
        return new StringBuilder(value).reverse().toString();
    }
}
