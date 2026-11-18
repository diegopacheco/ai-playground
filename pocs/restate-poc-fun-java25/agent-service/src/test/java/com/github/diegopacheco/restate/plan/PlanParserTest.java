package com.github.diegopacheco.restate.plan;

import com.github.diegopacheco.restate.AgentException;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PlanParserTest {

    @Test
    void extractsTheTaskListEvenWhenTheModelWrapsItInChatter() {
        Plan plan = PlanParser.parse("Sure! Here is the plan:\n```json\n[\"find facts\", \"  pick three \"]\n```");
        assertEquals(List.of("find facts", "pick three"), plan.tasks());
    }

    @Test
    void capsThePlanSoARunawayModelCannotSpawnUnboundedPaidSteps() {
        Plan plan = PlanParser.parse("[\"a\",\"b\",\"c\",\"d\",\"e\",\"f\"]");
        assertEquals(PlanParser.MAX_TASKS, plan.tasks().size());
    }

    @Test
    void dropsBlankAndNonTextEntriesSoEveryStepHasRealWork() {
        Plan plan = PlanParser.parse("[\"\", 3, \"write\", null]");
        assertEquals(List.of("write"), plan.tasks());
    }

    @Test
    void failsWhenThereIsNoListSoTheDurableStepRetriesInsteadOfJournalingGarbage() {
        AgentException error = assertThrows(AgentException.class, () -> PlanParser.parse("I cannot help with that"));
        assertTrue(error.getMessage().contains("no JSON task list"));
    }

    @Test
    void failsOnAnEmptyPlanBecauseARunWithoutTasksWouldSummarizeNothing() {
        assertThrows(AgentException.class, () -> PlanParser.parse("[\"  \"]"));
    }
}
