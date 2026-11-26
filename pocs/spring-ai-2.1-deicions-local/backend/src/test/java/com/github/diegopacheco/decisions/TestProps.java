package com.github.diegopacheco.decisions;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;

import com.github.diegopacheco.decisions.config.DecisionsProperties;

public final class TestProps {

    public static final DecisionsProperties PROPS = new DecisionsProperties(0.9, 0.4, 2.0, 0.5);
    public static final Clock CLOCK = Clock.fixed(Instant.parse("2026-11-26T00:00:00Z"), ZoneOffset.UTC);

    private TestProps() {
    }
}
