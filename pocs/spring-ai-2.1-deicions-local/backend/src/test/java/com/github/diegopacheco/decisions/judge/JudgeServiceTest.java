package com.github.diegopacheco.decisions.judge;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import com.github.diegopacheco.decisions.FakeDecisionModel;
import com.github.diegopacheco.decisions.TestProps;

import org.junit.jupiter.api.Test;

class JudgeServiceTest {

    private JudgeService judgeWith(double plausible, List<Double> helpfulness) {
        FakeDecisionModel model = new FakeDecisionModel(name -> plausible, helpfulness);
        return new JudgeService(new JudgeConfig().decisionJudge(model, TestProps.PROPS), TestProps.PROPS, TestProps.CLOCK);
    }

    @Test
    void anImpossibleTemperatureFailsAndTellsTheChatModelWhatToFix() {
        JudgeResult result = judgeWith(0.21, List.of(0.0, 0.0, 0.1, 0.9)).judge("Weather in Paris?", "Sunny, -255°C");

        assertFalse(result.passed());
        assertTrue(result.feedback().contains("is_plausible"));
        JudgeResult.Finding plausible = result.findings().stream().filter(f -> f.name().equals("is_plausible"))
                .findFirst().orElseThrow();
        assertEquals("FAILED", plausible.outcome());
    }

    @Test
    void anUnhelpfulAnswerFailsEvenWhenItsNumbersArePlausible() {
        JudgeResult result = judgeWith(0.95, List.of(0.9, 0.1, 0.0, 0.0)).judge("Weather in Paris?", "I like trains");

        assertFalse(result.passed());
        assertTrue(result.feedback().contains("helpfulness"));
    }

    @Test
    void helpfulnessPassesOnProbabilityMassNotOnTheExpectedLevel() {
        JudgeResult result = judgeWith(0.95, List.of(0.05, 0.2, 0.6, 0.15)).judge("Weather in Paris?", "Sunny");
        JudgeResult.Finding helpfulness = result.findings().stream().filter(f -> f.name().equals("helpfulness"))
                .findFirst().orElseThrow();

        assertEquals("PASSED", helpfulness.outcome());
        assertEquals(0.75, helpfulness.value(), 1e-9);
        assertEquals(0.5, helpfulness.threshold());
        assertTrue(helpfulness.measure().contains("expected level 1.85"));
    }

    @Test
    void aHelpfulPlausibleAnswerPassesWithNoFeedback() {
        JudgeResult result = judgeWith(0.95, List.of(0.0, 0.0, 0.1, 0.9)).judge("Weather in Paris?", "Sunny, 15°C");

        assertTrue(result.passed());
        assertEquals("", result.feedback());
        assertEquals(2, result.findings().size());
    }
}
