package com.github.diegopacheco.decisions;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.DecisionResponseMetadata;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.answer.DecisionAnswer;
import org.springframework.ai.decision.answer.ScoreAnswer;
import org.springframework.ai.decision.answer.YesNoAnswer;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

public final class FakeDecisionModel implements DecisionModel {

    public static final String MODEL = "fake-decider";

    private final Function<String, Double> yesProbability;
    private final List<Double> scoreProbabilities;
    private DecisionPrompt lastPrompt;

    public FakeDecisionModel(Function<String, Double> yesProbability, List<Double> scoreProbabilities) {
        this.yesProbability = yesProbability;
        this.scoreProbabilities = scoreProbabilities;
    }

    public DecisionPrompt lastPrompt() {
        return lastPrompt;
    }

    @Override
    public DecisionResponse call(DecisionPrompt prompt) {
        this.lastPrompt = prompt;
        DecisionResponse.Builder builder = DecisionResponse.builder(prompt)
                .metadata(DecisionResponseMetadata.builder().model(MODEL).build());
        prompt.getQuestions().forEach((name, question) -> builder.answer(name, answer(name, question)));
        return builder.build();
    }

    private DecisionAnswer answer(String name, DecisionQuestion question) {
        return switch (question) {
            case YesNoQuestion q -> YesNoAnswer.builder().probability(yesProbability.apply(name)).build();
            case ChoiceQuestion q -> ChoiceAnswer.builder().value(q.getOptionNames().getFirst())
                    .probabilities(Map.of(q.getOptionNames().getFirst(), 1.0)).build();
            case ScoreQuestion q -> score(q.getLevels().size());
            default -> throw new IllegalArgumentException("unsupported question " + name);
        };
    }

    private ScoreAnswer score(int levels) {
        List<Double> probabilities = scoreProbabilities.size() == levels ? scoreProbabilities
                : java.util.stream.IntStream.range(0, levels).mapToObj(i -> i == 0 ? 1.0 : 0.0).toList();
        double expected = java.util.stream.IntStream.range(0, levels).mapToDouble(i -> i * probabilities.get(i)).sum();
        return ScoreAnswer.builder().value(expected).probabilities(probabilities).build();
    }
}
