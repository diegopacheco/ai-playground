package com.github.diegopacheco.decisions.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("decisions")
public record DecisionsProperties(double autoThreshold, double reviewThreshold, double helpfulnessMinLevel,
        double plausibleMinProbability) {
}
