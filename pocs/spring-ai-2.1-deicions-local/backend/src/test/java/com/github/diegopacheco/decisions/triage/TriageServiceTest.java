package com.github.diegopacheco.decisions.triage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.List;
import java.util.Set;

import com.github.diegopacheco.decisions.FakeDecisionModel;
import com.github.diegopacheco.decisions.TestProps;

import org.junit.jupiter.api.Test;

class TriageServiceTest {

    @Test
    void aNoAtFortyFivePercentIsRoutedToReviewNotDropped() {
        FakeDecisionModel model = new FakeDecisionModel(name -> 0.45, List.of(0.7, 0.2, 0.1));
        TriageResult result = new TriageService(model, TestProps.PROPS, TestProps.CLOCK).triage("Where is my order?");

        assertFalse(result.defect().yes());
        assertEquals(Route.HUMAN_REVIEW, result.route());
        assertEquals(0.45, result.defect().probability());
        assertEquals(new TriageResult.Thresholds(0.9, 0.4), result.thresholds());
    }

    @Test
    void oneCallAsksAllThreeTypedQuestionsAboutTheSameMessage() {
        FakeDecisionModel model = new FakeDecisionModel(name -> 0.95, List.of(0.1, 0.1, 0.8));
        TriageResult result = new TriageService(model, TestProps.PROPS, TestProps.CLOCK).triage("left side is dead");

        assertEquals(Set.of("defect", "topic", "satisfaction"), model.lastPrompt().getQuestions().keySet());
        assertEquals(Route.AUTO_REPAIR_TICKET, result.route());
        assertEquals("happy", result.satisfaction().label());
        assertEquals(FakeDecisionModel.MODEL, result.model());
    }
}
