package com.github.diegopacheco.decisions.judge;

import java.util.List;

public record JudgeResult(long latencyMs, boolean passed, List<Finding> findings, String feedback) {

    public record Finding(String name, String outcome, String measure, double value, double threshold, String detail) {
    }
}
