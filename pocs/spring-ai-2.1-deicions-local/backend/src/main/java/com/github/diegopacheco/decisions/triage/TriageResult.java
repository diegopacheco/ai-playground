package com.github.diegopacheco.decisions.triage;

import java.util.List;
import java.util.Map;

public record TriageResult(String model, long latencyMs, Defect defect, Topic topic, Satisfaction satisfaction,
        Route route, Thresholds thresholds) {

    public record Thresholds(double auto, double review) {
    }

    public record Defect(boolean yes, double probability) {
    }

    public record Topic(String value, Map<String, Double> probabilities) {
    }

    public record Satisfaction(String label, int nearestLevel, double expectedLevel, List<String> levels,
            List<Double> probabilities) {
    }
}
