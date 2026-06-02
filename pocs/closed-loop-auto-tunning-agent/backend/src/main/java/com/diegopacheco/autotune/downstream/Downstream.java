package com.diegopacheco.autotune.downstream;

import org.springframework.stereotype.Component;

import java.util.concurrent.ThreadLocalRandom;

@Component
public class Downstream {

    private final ScenarioState scenario;

    public Downstream(ScenarioState scenario) {
        this.scenario = scenario;
    }

    public String call() {
        long base = scenario.latencyMs();
        long jitter = scenario.jitterMs();
        long sleep = base;
        if (jitter > 0) {
            sleep = base + ThreadLocalRandom.current().nextLong(-jitter, jitter + 1);
        }
        if (sleep > 0) {
            try {
                Thread.sleep(sleep);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("downstream interrupted", e);
            }
        }
        if (ThreadLocalRandom.current().nextDouble() < scenario.failRate()) {
            throw new RuntimeException("downstream failure");
        }
        return "ok";
    }
}
