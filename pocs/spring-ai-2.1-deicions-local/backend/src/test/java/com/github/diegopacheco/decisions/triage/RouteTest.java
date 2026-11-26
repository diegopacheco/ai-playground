package com.github.diegopacheco.decisions.triage;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class RouteTest {

    @Test
    void aVerySureDefectOpensARepairTicketWithoutAHuman() {
        assertEquals(Route.AUTO_REPAIR_TICKET, Route.of(0.95, 0.9, 0.4));
        assertEquals(Route.AUTO_REPAIR_TICKET, Route.of(0.9, 0.9, 0.4));
    }

    @Test
    void aCoinFlipNoStillGoesToAHumanBecauseTheBooleanHidesTheDoubt() {
        assertEquals(Route.HUMAN_REVIEW, Route.of(0.45, 0.9, 0.4));
        assertEquals(Route.HUMAN_REVIEW, Route.of(0.4, 0.9, 0.4));
        assertEquals(Route.HUMAN_REVIEW, Route.of(0.89, 0.9, 0.4));
    }

    @Test
    void aConfidentNoNeedsNoAction() {
        assertEquals(Route.NO_ACTION, Route.of(0.02, 0.9, 0.4));
        assertEquals(Route.NO_ACTION, Route.of(0.39, 0.9, 0.4));
    }
}
