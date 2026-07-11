package com.diegopacheco.temporalpoc.workflow;

import com.diegopacheco.temporalpoc.service.CodexCliService;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class StockAgentActivityImpl implements StockAgentActivity {
    private static final Logger log = LoggerFactory.getLogger(StockAgentActivityImpl.class);
    private final CodexCliService codex;

    public StockAgentActivityImpl(CodexCliService codex) {
        this.codex = codex;
    }

    @Override
    public String researchStock(String symbol, String company) {
        ActivityInfo info = Activity.getExecutionContext().getInfo();
        log.info("stock activity started workflowId={} runId={} activityId={} attempt={} symbol={} company={}", info.getWorkflowId(), info.getRunId(), info.getActivityId(), info.getAttempt(), symbol, company);
        String result = codex.ask("You are the stock research agent for a Java Temporal workflow. Do not inspect files. Do not run commands. Do not ask questions. For " + company + " stock symbol " + symbol + ", provide a concise stock summary with fundamentals, valuation, recent price movement, and analyst sentiment. If live market data is unavailable, say that clearly. Return at most 8 bullets.");
        log.info("stock activity completed workflowId={} activityId={} attempt={} resultLength={}", info.getWorkflowId(), info.getActivityId(), info.getAttempt(), result.length());
        return result;
    }
}
