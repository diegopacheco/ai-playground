package com.diegopacheco.temporalpoc.workflow;

import com.diegopacheco.temporalpoc.domain.Decision;
import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.workflow.Workflow;
import org.slf4j.Logger;

import java.time.Duration;

public class CompanyResearchWorkflowImpl implements CompanyResearchWorkflow {
    private static final Logger log = Workflow.getLogger(CompanyResearchWorkflowImpl.class);
    private final ActivityOptions options = ActivityOptions.newBuilder()
            .setStartToCloseTimeout(Duration.ofSeconds(40))
            .setRetryOptions(RetryOptions.newBuilder()
                    .setInitialInterval(Duration.ofSeconds(1))
                    .setMaximumInterval(Duration.ofSeconds(5))
                    .setMaximumAttempts(3)
                    .build())
            .build();
    private final StockAgentActivity stock = Workflow.newActivityStub(StockAgentActivity.class, options);
    private final NewsAgentActivity news = Workflow.newActivityStub(NewsAgentActivity.class, options);
    private final DecisionAgentActivity decision = Workflow.newActivityStub(DecisionAgentActivity.class, options);

    @Override
    public ResearchResult research(String symbol, String company) {
        log.info("workflow step=start workflowId={} runId={} symbol={} company={}", Workflow.getInfo().getWorkflowId(), Workflow.getInfo().getRunId(), symbol, company);
        log.info("workflow step=stock_research_start message=calling_stock_agent symbol={} company={}", symbol, company);
        String stockSummary = stock.researchStock(symbol, company);
        log.info("workflow step=stock_research_done message=stock_agent_returned symbol={} stockLength={}", symbol, stockSummary.length());
        log.info("workflow step=news_research_start message=calling_news_agent symbol={} company={}", symbol, company);
        String newsSummary = news.researchNews(symbol, company);
        log.info("workflow step=news_research_done message=news_agent_returned symbol={} newsLength={}", symbol, newsSummary.length());
        log.info("workflow step=decision_start message=calling_decision_agent symbol={} stockLength={} newsLength={}", symbol, stockSummary.length(), newsSummary.length());
        Decision result = decision.decide(symbol, company, stockSummary, newsSummary);
        log.info("workflow step=decision_done message=decision_agent_returned symbol={} recommendation={} confidence={} rationaleLength={}", symbol, result.recommendation(), result.confidence(), result.rationale().length());
        log.info("workflow step=complete workflowId={} runId={} symbol={} recommendation={} confidence={}", Workflow.getInfo().getWorkflowId(), Workflow.getInfo().getRunId(), symbol, result.recommendation(), result.confidence());
        return new ResearchResult(stockSummary, newsSummary, result);
    }
}
