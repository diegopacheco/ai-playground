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
        String result = codex.ask("Research current stock fundamentals, valuation, recent price movement, and analyst sentiment for " + company + " stock symbol " + symbol + ". Return concise bullets.");
        log.info("stock activity completed workflowId={} activityId={} attempt={} resultLength={}", info.getWorkflowId(), info.getActivityId(), info.getAttempt(), result.length());
        return result;
    }
}
