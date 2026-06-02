package com.diegopacheco.autotune.pattern;

import com.diegopacheco.autotune.config.CircuitBreakerSettings;
import com.diegopacheco.autotune.metrics.CbMetrics;
import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterConfig;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

@Component
public class BreakerManager {

    private static final String CB_NAME = "downstream";

    private final CircuitBreakerRegistry cbRegistry = CircuitBreakerRegistry.ofDefaults();
    private final AtomicReference<CircuitBreaker> cbRef = new AtomicReference<>();
    private volatile CircuitBreakerSettings current;

    private final Retry retry = Retry.of("retry", RetryConfig.custom()
            .maxAttempts(3)
            .waitDuration(Duration.ofMillis(200))
            .build());

    private final RateLimiter rateLimiter = RateLimiter.of("ratelimiter", RateLimiterConfig.custom()
            .limitForPeriod(5)
            .limitRefreshPeriod(Duration.ofSeconds(1))
            .timeoutDuration(Duration.ZERO)
            .build());

    private final Bulkhead bulkhead = Bulkhead.of("bulkhead", BulkheadConfig.custom()
            .maxConcurrentCalls(5)
            .maxWaitDuration(Duration.ZERO)
            .build());

    public BreakerManager() {
        apply(defaults());
    }

    public static CircuitBreakerSettings defaults() {
        return new CircuitBreakerSettings(50, 100, 2000, "COUNT", 10, 5, 10, 3);
    }

    public synchronized void apply(CircuitBreakerSettings s) {
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
                .failureRateThreshold((float) s.failureRateThreshold())
                .slowCallRateThreshold((float) s.slowCallRateThreshold())
                .slowCallDurationThreshold(Duration.ofMillis(s.slowCallDurationThresholdMs()))
                .slidingWindowType("TIME".equalsIgnoreCase(s.slidingWindowType())
                        ? CircuitBreakerConfig.SlidingWindowType.TIME_BASED
                        : CircuitBreakerConfig.SlidingWindowType.COUNT_BASED)
                .slidingWindowSize(s.slidingWindowSize())
                .minimumNumberOfCalls(s.minimumNumberOfCalls())
                .waitDurationInOpenState(Duration.ofSeconds(s.waitDurationInOpenStateSeconds()))
                .permittedNumberOfCallsInHalfOpenState(s.permittedNumberOfCallsInHalfOpenState())
                .automaticTransitionFromOpenToHalfOpenEnabled(true)
                .build();
        cbRegistry.remove(CB_NAME);
        cbRef.set(cbRegistry.circuitBreaker(CB_NAME, config));
        this.current = s;
    }

    public void reset() {
        cb().reset();
    }

    public CircuitBreaker cb() {
        return cbRef.get();
    }

    public Retry retry() {
        return retry;
    }

    public RateLimiter rateLimiter() {
        return rateLimiter;
    }

    public Bulkhead bulkhead() {
        return bulkhead;
    }

    public CircuitBreakerSettings currentSettings() {
        return current;
    }

    public CbMetrics cbMetrics() {
        CircuitBreaker cb = cb();
        CircuitBreaker.Metrics m = cb.getMetrics();
        return new CbMetrics(
                cb.getState().name(),
                m.getFailureRate(),
                m.getSlowCallRate(),
                m.getNumberOfBufferedCalls(),
                m.getNumberOfFailedCalls(),
                m.getNumberOfSlowCalls(),
                m.getNumberOfSuccessfulCalls(),
                m.getNumberOfNotPermittedCalls(),
                System.currentTimeMillis());
    }
}
