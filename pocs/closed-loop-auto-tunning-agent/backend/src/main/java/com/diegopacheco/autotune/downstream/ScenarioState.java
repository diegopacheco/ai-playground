package com.diegopacheco.autotune.downstream;

import org.springframework.stereotype.Component;

@Component
public class ScenarioState {

    private volatile double failRate = 0.5;
    private volatile long latencyMs = 200;
    private volatile long jitterMs = 100;

    public Scenario get() {
        return new Scenario(failRate, latencyMs, jitterMs);
    }

    public void set(Scenario s) {
        this.failRate = Math.max(0.0, Math.min(1.0, s.failRate()));
        this.latencyMs = Math.max(0, s.latencyMs());
        this.jitterMs = Math.max(0, s.jitterMs());
    }

    public double failRate() {
        return failRate;
    }

    public long latencyMs() {
        return latencyMs;
    }

    public long jitterMs() {
        return jitterMs;
    }
}
