package com.diegopacheco.temporalpoc.workflow;

import com.diegopacheco.temporalpoc.domain.Decision;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.testing.TestWorkflowEnvironment;
import io.temporal.worker.Worker;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class CompanyResearchWorkflowTest {
    @Test
    void shouldRunAgentPipeline() {
        TestWorkflowEnvironment environment = TestWorkflowEnvironment.newInstance();
        Worker worker = environment.newWorker("test-queue");
        worker.registerWorkflowImplementationTypes(CompanyResearchWorkflowImpl.class);
        worker.registerActivitiesImplementations(
                new StockAgentActivity() {
                    @Override
                    public String researchStock(String symbol, String company) {
                        return "Revenue growth and healthy cash flow.";
                    }
                },
                new NewsAgentActivity() {
                    @Override
                    public String researchNews(String symbol, String company) {
                        return "Latest news is neutral.";
                    }
                },
                new DecisionAgentActivity() {
                    @Override
                    public Decision decide(String symbol, String company, String stockSummary, String newsSummary) {
                        return new Decision("HOLD", 60, stockSummary + " " + newsSummary);
                    }
                }
        );
        environment.start();
        WorkflowClient client = environment.getWorkflowClient();
        CompanyResearchWorkflow workflow = client.newWorkflowStub(
                CompanyResearchWorkflow.class,
                WorkflowOptions.newBuilder().setTaskQueue("test-queue").build()
        );
        ResearchResult result = workflow.research("AAPL", "Apple");
        assertThat(result.decision().recommendation()).isEqualTo("HOLD");
        assertThat(result.stockSummary()).contains("Revenue");
        environment.close();
    }
}
