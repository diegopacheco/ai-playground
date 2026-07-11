package com.diegopacheco.temporalpoc.workflow;

import com.diegopacheco.temporalpoc.domain.Decision;
import io.temporal.activity.ActivityInterface;

@ActivityInterface
public interface DecisionAgentActivity {
    Decision decide(String symbol, String company, String stockSummary, String newsSummary);
}
