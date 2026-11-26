package com.github.diegopacheco.decisions.web;

import java.time.Clock;
import java.util.function.Supplier;

public final class Stopwatch {

    private Stopwatch() {
    }

    public static <T> Timed<T> time(Clock clock, Supplier<T> call) {
        long start = clock.millis();
        T value = call.get();
        return new Timed<>(value, clock.millis() - start);
    }

    public record Timed<T>(T value, long millis) {
    }
}
