package com.diegopacheco.temporalpoc.workflow;

import com.diegopacheco.temporalpoc.service.CodexCliService;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class NewsAgentActivityImpl implements NewsAgentActivity {
    private static final Logger log = LoggerFactory.getLogger(NewsAgentActivityImpl.class);
    private final CodexCliService codex;

    public NewsAgentActivityImpl(CodexCliService codex) {
        this.codex = codex;
    }

    @Override
    public String researchNews(String symbol, String company) {
        ActivityInfo info = Activity.getExecutionContext().getInfo();
        log.info("news activity started workflowId={} runId={} activityId={} attempt={} symbol={} company={}", info.getWorkflowId(), info.getRunId(), info.getActivityId(), info.getAttempt(), symbol, company);
        String result = codex.ask("You are the news research agent for a Java Temporal workflow. Do not inspect files. Do not run commands. Do not ask questions. For " + company + " stock symbol " + symbol + ", summarize latest material company news relevant to investors. If live news is unavailable, say that clearly. Return at most 8 bullets with dates when available.");
        log.info("news activity completed workflowId={} activityId={} attempt={} resultLength={}", info.getWorkflowId(), info.getActivityId(), info.getAttempt(), result.length());
        return result;
    }
}
