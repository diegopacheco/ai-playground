package com.diegopacheco.autotune.pattern;

import com.diegopacheco.autotune.downstream.Downstream;
import io.github.resilience4j.bulkhead.Bulkhead;
import io.github.resilience4j.bulkhead.BulkheadFullException;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RequestNotPermitted;
import io.github.resilience4j.retry.Retry;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class PatternController {

    private final Downstream downstream;
    private final BreakerManager manager;

    public PatternController(Downstream downstream, BreakerManager manager) {
        this.downstream = downstream;
        this.manager = manager;
    }

    @PostMapping("/cb/call")
    public CallResult cb() {
        long start = System.nanoTime();
        CircuitBreaker cb = manager.cb();
        try {
            cb.executeSupplier(downstream::call);
            return new CallResult("SUCCESS", elapsed(start));
        } catch (CallNotPermittedException e) {
            return new CallResult("SHORT_CIRCUITED", elapsed(start));
        } catch (RuntimeException e) {
            return new CallResult("FAILURE", elapsed(start));
        }
    }

    @PostMapping("/retry/call")
    public CallResult retry() {
        long start = System.nanoTime();
        try {
            Retry.decorateSupplier(manager.retry(), downstream::call).get();
            return new CallResult("SUCCESS", elapsed(start));
        } catch (RuntimeException e) {
            return new CallResult("FAILURE", elapsed(start));
        }
    }

    @PostMapping("/ratelimiter/call")
    public CallResult rateLimiter() {
        long start = System.nanoTime();
        try {
            RateLimiter.decorateSupplier(manager.rateLimiter(), downstream::call).get();
            return new CallResult("SUCCESS", elapsed(start));
        } catch (RequestNotPermitted e) {
            return new CallResult("RATE_LIMITED", elapsed(start));
        } catch (RuntimeException e) {
            return new CallResult("FAILURE", elapsed(start));
        }
    }

    @PostMapping("/bulkhead/call")
    public CallResult bulkhead() {
        long start = System.nanoTime();
        try {
            Bulkhead.decorateSupplier(manager.bulkhead(), downstream::call).get();
            return new CallResult("SUCCESS", elapsed(start));
        } catch (BulkheadFullException e) {
            return new CallResult("REJECTED", elapsed(start));
        } catch (RuntimeException e) {
            return new CallResult("FAILURE", elapsed(start));
        }
    }

    @PostMapping("/cb/reset")
    public void reset() {
        manager.reset();
    }

    private static long elapsed(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000;
    }
}
