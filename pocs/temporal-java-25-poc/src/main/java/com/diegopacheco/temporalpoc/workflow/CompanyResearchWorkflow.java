package com.diegopacheco.temporalpoc.workflow;

import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;

@WorkflowInterface
public interface CompanyResearchWorkflow {
    @WorkflowMethod
    ResearchResult research(String symbol, String company);
}
