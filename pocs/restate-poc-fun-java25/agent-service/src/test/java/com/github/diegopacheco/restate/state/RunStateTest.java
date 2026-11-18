package com.github.diegopacheco.restate.state;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RunStateTest {

    private final RunState planned = RunState.start("goal", "claude", "m").withPlan(List.of("a", "b"));

    @Test
    void aPlanMovesTheRunToWorkingWithEveryStepPending() {
        assertEquals(Phase.WORKING, planned.phase());
        assertTrue(planned.steps().stream().allMatch(step -> step.status() == StepStatus.PENDING));
    }

    @Test
    void onlyFinishedStepsFeedTheNextWorkerSoItNeverSeesUnfinishedWork() {
        RunState state = planned.stepDone(0, "out a").stepRunning(1);
        assertEquals("Task 1: a\nResult: out a", state.completedWork());
    }

    @Test
    void failingKeepsTheFinishedStepsSoTheUserSeesHowFarTheRunGot() {
        RunState state = planned.stepDone(0, "out a").failed("boom");
        assertEquals(Phase.FAILED, state.phase());
        assertEquals("boom", state.error());
        assertEquals(StepStatus.DONE, state.steps().getFirst().status());
    }

    @Test
    void transitionsNeverMutateTheJournaledState() {
        planned.stepDone(0, "x");
        assertEquals(StepStatus.PENDING, planned.steps().getFirst().status());
    }

    @Test
    void serializesTheLowercaseContractTheUiReads() throws Exception {
        String json = new ObjectMapper().writeValueAsString(planned.stepRunning(0));
        assertTrue(json.contains("\"phase\":\"working\""));
        assertTrue(json.contains("\"status\":\"running\""));
        assertEquals(planned.stepRunning(0), new ObjectMapper().readValue(json, RunState.class));
    }
}
