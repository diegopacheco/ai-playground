package com.diegopacheco.temporalpoc.service;

import com.diegopacheco.temporalpoc.api.TriggerResponse;
import com.diegopacheco.temporalpoc.domain.Decision;
import com.diegopacheco.temporalpoc.domain.ResearchReport;
import com.diegopacheco.temporalpoc.repository.ResearchReportRepository;
import com.diegopacheco.temporalpoc.workflow.CompanyResearchWorkflow;
import com.diegopacheco.temporalpoc.workflow.ResearchResult;
import io.temporal.api.common.v1.WorkflowExecution;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

import java.time.OffsetDateTime;
import java.util.UUID;

@Service
public class ResearchService {
    private static final Logger log = LoggerFactory.getLogger(ResearchService.class);
    private final WorkflowClient client;
    private final ResearchReportRepository repository;
    private final String taskQueue;

    public ResearchService(WorkflowClient client, ResearchReportRepository repository, @Value("${temporal.task-queue}") String taskQueue) {
        this.client = client;
        this.repository = repository;
        this.taskQueue = taskQueue;
    }

    public ResearchReport research(String symbol, String company) {
        log.info("blocking workflow start requested symbol={} company={}", symbol, company);
        CompanyResearchWorkflow workflow = workflow(symbol);
        ResearchResult result = workflow.research(symbol, company);
        log.info("blocking workflow completed symbol={} company={} stockLength={} newsLength={} recommendation={} confidence={}", symbol, company, result.stockSummary().length(), result.newsSummary().length(), result.decision().recommendation(), result.decision().confidence());
        Decision decision = result.decision();
        ResearchReport report = new ResearchReport(
                null,
                symbol,
                company,
                result.stockSummary(),
                result.newsSummary(),
                decision.recommendation(),
                decision.confidence(),
                decision.rationale(),
                OffsetDateTime.now()
        );
        ResearchReport saved = repository.save(report);
        log.info("research report saved id={} symbol={} company={} recommendation={} confidence={}", saved.id(), saved.symbol(), saved.company(), saved.recommendation(), saved.confidence());
        return saved;
    }

    public TriggerResponse trigger(String symbol, String company) {
        String workflowId = workflowId(symbol);
        log.info("async workflow start requested workflowId={} symbol={} company={} taskQueue={}", workflowId, symbol, company, taskQueue);
        CompanyResearchWorkflow workflow = client.newWorkflowStub(
                CompanyResearchWorkflow.class,
                WorkflowOptions.newBuilder()
                        .setTaskQueue(taskQueue)
                        .setWorkflowId(workflowId)
                        .build()
        );
        WorkflowExecution execution = WorkflowClient.start(workflow::research, symbol, company);
        String temporalUrl = "http://localhost:8081/namespaces/default/workflows/" + workflowId + "/" + execution.getRunId() + "/timeline";
        log.info("async workflow started workflowId={} runId={} temporalUrl={}", workflowId, execution.getRunId(), temporalUrl);
        return new TriggerResponse(workflowId, execution.getRunId(), temporalUrl);
    }

    public Page<ResearchReport> reports(int page, int size) {
        log.info("loading research reports page={} size={}", page, size);
        Page<ResearchReport> reports = repository.findAll(PageRequest.of(page, size));
        log.info("loaded research reports page={} size={} totalElements={} totalPages={}", reports.getNumber(), reports.getSize(), reports.getTotalElements(), reports.getTotalPages());
        return reports;
    }

    private CompanyResearchWorkflow workflow(String symbol) {
        String workflowId = workflowId(symbol);
        log.info("creating blocking workflow stub workflowId={} taskQueue={}", workflowId, taskQueue);
        return client.newWorkflowStub(
                CompanyResearchWorkflow.class,
                WorkflowOptions.newBuilder()
                        .setTaskQueue(taskQueue)
                        .setWorkflowId(workflowId)
                        .build()
        );
    }

    private String workflowId(String symbol) {
        return "company-research-" + symbol + "-" + UUID.randomUUID();
    }
}
