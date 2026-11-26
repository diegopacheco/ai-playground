package com.github.diegopacheco.decisions.triage;

import java.util.List;

import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

final class TriageQuestions {

    static final String DEFECT = "defect";
    static final String TOPIC = "topic";
    static final String SATISFACTION = "satisfaction";
    static final List<String> SATISFACTION_LEVELS = List.of("unhappy", "neutral", "happy");

    private TriageQuestions() {
    }

    static DecisionPrompt prompt(String message) {
        return DecisionPrompt.builder()
                .text(message)
                .question(DEFECT, YesNoQuestion.builder()
                        .text("Does the customer report a defect?")
                        .whenYes("Part of the item does not work as it should")
                        .whenNo("The item works, and the message is about something else")
                        .build())
                .question(TOPIC, ChoiceQuestion.builder()
                        .text("What is the message about?")
                        .option("delivery", "Where an order is, or when it will arrive")
                        .option("product", "A problem with an item the customer received")
                        .option("account", "Login, address or payment details")
                        .option("other", "None of the above")
                        .build())
                .question(SATISFACTION, ScoreQuestion.of("How satisfied is the customer?",
                        SATISFACTION_LEVELS.toArray(String[]::new)))
                .build();
    }
}
