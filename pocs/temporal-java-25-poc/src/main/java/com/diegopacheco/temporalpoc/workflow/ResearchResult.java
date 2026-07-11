package com.diegopacheco.temporalpoc.workflow;

import com.diegopacheco.temporalpoc.domain.Decision;

public record ResearchResult(String stockSummary, String newsSummary, Decision decision) {
}
