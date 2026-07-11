package com.diegopacheco.temporalpoc.api;

public record TriggerResponse(String workflowId, String runId, String temporalUrl) {
}
