package com.store.util;

public final class MoneyUtils {

    private MoneyUtils() {
    }

    public static long applyRate(long cents, double rate) {
        return Math.round(cents * rate);
    }

    public static long toCents(double amount) {
        return Math.round(amount * 100);
    }
}
