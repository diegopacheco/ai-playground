package com.diegopacheco.temporalpoc.workflow;

import com.diegopacheco.temporalpoc.domain.Decision;
import com.diegopacheco.temporalpoc.service.CodexCliService;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class DecisionAgentActivityImpl implements DecisionAgentActivity {
    private static final Logger log = LoggerFactory.getLogger(DecisionAgentActivityImpl.class);
    private final CodexCliService codex;

    public DecisionAgentActivityImpl(CodexCliService codex) {
        this.codex = codex;
    }

    @Override
    public Decision decide(String symbol, String company, String stockSummary, String newsSummary) {
        ActivityInfo info = Activity.getExecutionContext().getInfo();
        log.info("decision activity started workflowId={} runId={} activityId={} attempt={} symbol={} company={} stockLength={} newsLength={}", info.getWorkflowId(), info.getRunId(), info.getActivityId(), info.getAttempt(), symbol, company, stockSummary.length(), newsSummary.length());
        String prompt = """
                You are the decision agent for a Java Temporal workflow. Do not inspect files. Do not run commands. Do not ask questions. Decide BUY or HOLD for %s %s from the provided stock and news summaries only.
                Stock research:
                %s
                News research:
                %s
                Return one line beginning with BUY or HOLD, then a concise rationale.
                """.formatted(symbol, company, stockSummary, newsSummary);
        String rationale = codex.ask(prompt);
        String upper = rationale.toUpperCase();
        String recommendation = upper.contains("BUY") && !upper.contains("HOLD") ? "BUY" : "HOLD";
        int confidence = upper.contains("STRONG") ? 80 : 60;
        log.info("decision activity completed workflowId={} activityId={} attempt={} recommendation={} confidence={} rationaleLength={}", info.getWorkflowId(), info.getActivityId(), info.getAttempt(), recommendation, confidence, rationale.length());
        return new Decision(recommendation, confidence, rationale);
    }
}
