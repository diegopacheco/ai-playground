package com.github.diegopacheco.adminconsole.trace;

public record TraceBudget(int sourcesPerConnection, int rowsPerSource, int scanWindow, int totalHits,
                          long perConnectionTimeoutSeconds) {
    public static TraceBudget defaults() {
        return new TraceBudget(12, 20, 500, 200, 5);
    }
}
