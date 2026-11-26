package com.github.diegopacheco.decisions.guardrail;

import java.time.Clock;
import java.util.Map;

import com.github.diegopacheco.decisions.logging.AppLogger;
import com.github.diegopacheco.decisions.web.Stopwatch;

import org.springaicommunity.decision.guardrail.DecisionGuardrail;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.stereotype.Service;

@Service
public class GuardrailService {

    private static final AppLogger log = AppLogger.of(GuardrailService.class);

    private final DecisionModel decisionModel;
    private final Clock clock;

    public GuardrailService(DecisionModel decisionModel, Clock clock) {
        this.decisionModel = decisionModel;
        this.clock = clock;
    }

    public GuardrailResult screen(Battery battery, String text) {
        Stopwatch.Timed<DecisionGuardrail.Verdict> timed = Stopwatch.time(clock,
                () -> battery.guardrail().screen(decisionModel, text));
        DecisionGuardrail.Verdict verdict = timed.value();
        log.info("guardrail screened", Map.of("battery", battery, "outcome", verdict.outcome(), "ms", timed.millis()));
        return new GuardrailResult(battery, timed.millis(), verdict.outcome().name(), verdict.blocked(),
                verdict.severity(), verdict.scores(), verdict.triggered(), verdict.flagged(), verdict.summary());
    }
}
