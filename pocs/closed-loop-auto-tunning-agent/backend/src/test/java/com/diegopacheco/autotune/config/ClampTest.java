package com.diegopacheco.autotune.config;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ClampTest {

    @Test
    void adversarialProposalCannotDisableProtection() {
        CircuitBreakerSettings hostile = new CircuitBreakerSettings(
                100, 0, 1, "COUNT", 1, 1, 0, 0);

        Clamp.Result r = Clamp.apply(hostile);
        CircuitBreakerSettings s = r.settings();

        assertThat(s.failureRateThreshold()).isBetween(40.0, 70.0);
        assertThat(s.slowCallRateThreshold()).isBetween(50.0, 100.0);
        assertThat(s.slowCallDurationThresholdMs()).isBetween(200L, 5000L);
        assertThat(s.slidingWindowSize()).isBetween(10, 200);
        assertThat(s.minimumNumberOfCalls()).isBetween(5, 100);
        assertThat(s.waitDurationInOpenStateSeconds()).isBetween(5L, 60L);
        assertThat(s.permittedNumberOfCallsInHalfOpenState()).isBetween(2, 20);

        assertThat(s.failureRateThreshold()).isEqualTo(70.0);
        assertThat(s.waitDurationInOpenStateSeconds()).isEqualTo(5L);
    }

    @Test
    void inRangeProposalIsUntouched() {
        CircuitBreakerSettings ok = new CircuitBreakerSettings(
                55, 80, 1500, "TIME", 50, 20, 20, 5);

        Clamp.Result r = Clamp.apply(ok);

        assertThat(r.settings()).isEqualTo(ok);
        assertThat(r.fields()).noneMatch(Clamp.FieldClamp::wasClamped);
    }

    @Test
    void clampedFieldsAreFlaggedWithProposedAndApplied() {
        CircuitBreakerSettings tooHigh = new CircuitBreakerSettings(
                95, 80, 1500, "COUNT", 50, 20, 120, 5);

        Clamp.Result r = Clamp.apply(tooHigh);

        Clamp.FieldClamp failure = r.fields().stream()
                .filter(f -> f.field().equals("failureRateThreshold")).findFirst().orElseThrow();
        assertThat(failure.wasClamped()).isTrue();
        assertThat(failure.proposed()).isEqualTo(95.0);
        assertThat(failure.applied()).isEqualTo(70.0);

        Clamp.FieldClamp slow = r.fields().stream()
                .filter(f -> f.field().equals("slowCallRateThreshold")).findFirst().orElseThrow();
        assertThat(slow.wasClamped()).isFalse();
    }
}
