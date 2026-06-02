package com.diegopacheco.autotune.metrics;

import com.diegopacheco.autotune.pattern.BreakerManager;
import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.retry.Retry;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/metrics")
public class MetricsController {

    private final BreakerManager manager;

    public MetricsController(BreakerManager manager) {
        this.manager = manager;
    }

    @GetMapping("/circuitbreaker")
    public CbMetrics circuitBreaker() {
        return manager.cbMetrics();
    }

    @GetMapping("/retry")
    public Map<String, Object> retry() {
        Retry.Metrics m = manager.retry().getMetrics();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("successWithoutRetry", m.getNumberOfSuccessfulCallsWithoutRetryAttempt());
        out.put("successWithRetry", m.getNumberOfSuccessfulCallsWithRetryAttempt());
        out.put("failedWithoutRetry", m.getNumberOfFailedCallsWithoutRetryAttempt());
        out.put("failedWithRetry", m.getNumberOfFailedCallsWithRetryAttempt());
        out.put("ts", System.currentTimeMillis());
        return out;
    }

    @GetMapping("/ratelimiter")
    public Map<String, Object> rateLimiter() {
        RateLimiter.Metrics m = manager.rateLimiter().getMetrics();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("availablePermissions", m.getAvailablePermissions());
        out.put("waitingThreads", m.getNumberOfWaitingThreads());
        out.put("ts", System.currentTimeMillis());
        return out;
    }

    @GetMapping("/bulkhead")
    public Map<String, Object> bulkhead() {
        Bulkhead.Metrics m = manager.bulkhead().getMetrics();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("availableConcurrentCalls", m.getAvailableConcurrentCalls());
        out.put("maxAllowedConcurrentCalls", m.getMaxAllowedConcurrentCalls());
        out.put("ts", System.currentTimeMillis());
        return out;
    }
}
