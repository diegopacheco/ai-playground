package com.diegopacheco.autotune.tune;

import com.diegopacheco.autotune.config.CircuitBreakerSettings;
import com.diegopacheco.autotune.config.Clamp;
import com.diegopacheco.autotune.metrics.CbMetrics;
import com.diegopacheco.autotune.pattern.BreakerManager;
import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class TuningService {

    private static final String SYSTEM = """
            You are a Resilience4j Circuit Breaker tuning advisor.
            SLO intent: serve as many requests successfully as possible while still tripping quickly when the
            downstream is genuinely unhealthy. Never mask real failures by disabling or neutering protection.
            Read the current configuration and the observed metrics, then propose an improved configuration.
            Respond with STRICT JSON only, no prose outside JSON, matching exactly these keys:
            {
              "failureRateThreshold": number (percent),
              "slowCallRateThreshold": number (percent),
              "slowCallDurationThresholdMs": integer,
              "slidingWindowType": "COUNT" or "TIME",
              "slidingWindowSize": integer,
              "minimumNumberOfCalls": integer,
              "waitDurationInOpenStateSeconds": integer,
              "permittedNumberOfCallsInHalfOpenState": integer,
              "rationale": string
            }
            Respect these hard bounds (values outside them will be clamped):
            failureRateThreshold 40-70, slowCallRateThreshold 50-100, slowCallDurationThresholdMs 200-5000,
            slidingWindowSize 10-200, minimumNumberOfCalls 5-100, waitDurationInOpenStateSeconds 5-60,
            permittedNumberOfCallsInHalfOpenState 2-20.
            The rationale must explain, in plain language, what the metrics indicate and why your changes help.
            """;

    private final OpenAiClient openai;
    private final BreakerManager manager;
    private final ObjectMapper mapper;

    public TuningService(OpenAiClient openai, BreakerManager manager, ObjectMapper mapper) {
        this.openai = openai;
        this.manager = manager;
        this.mapper = mapper;
    }

    public TuneResult tune(RunSummary run) {
        CircuitBreakerSettings current = manager.currentSettings();
        CbMetrics metrics = manager.cbMetrics();

        String user = mapper.writeValueAsString(buildContext(current, metrics, run));
        String content = openai.complete(SYSTEM, user);
        Proposal proposal = mapper.readValue(content, Proposal.class);

        CircuitBreakerSettings proposed = new CircuitBreakerSettings(
                proposal.failureRateThreshold(),
                proposal.slowCallRateThreshold(),
                proposal.slowCallDurationThresholdMs(),
                proposal.slidingWindowType(),
                proposal.slidingWindowSize(),
                proposal.minimumNumberOfCalls(),
                proposal.waitDurationInOpenStateSeconds(),
                proposal.permittedNumberOfCallsInHalfOpenState());

        Clamp.Result clamped = Clamp.apply(proposed);

        return new TuneResult(current, proposed, clamped.settings(), clamped.fields(),
                proposal.rationale(), openai.model(), metrics);
    }

    private Map<String, Object> buildContext(CircuitBreakerSettings current, CbMetrics metrics, RunSummary run) {
        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("currentConfig", current);
        ctx.put("liveMetrics", metrics);
        if (run != null) {
            ctx.put("observedRun", run);
        }
        return ctx;
    }
}
