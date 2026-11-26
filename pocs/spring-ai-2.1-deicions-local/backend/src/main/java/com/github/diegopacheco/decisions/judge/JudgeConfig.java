package com.github.diegopacheco.decisions.judge;

import com.github.diegopacheco.decisions.config.DecisionsProperties;

import org.springaicommunity.decision.judge.DecisionJudge;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class JudgeConfig {

    static final String HELPFULNESS = "helpfulness";
    static final String PLAUSIBLE = "is_plausible";

    @Bean
    DecisionJudge decisionJudge(DecisionModel decisionModel, DecisionsProperties props) {
        return DecisionJudge.builder(decisionModel)
                .score(HELPFULNESS, ScoreQuestion.builder()
                        .text("How well does `assistant_answer` address `user_question`?")
                        .level("Terrible: irrelevant or off-topic")
                        .level("Mostly unhelpful: misses the main point")
                        .level("Mostly helpful: minor gaps remain")
                        .level("Excellent: fully and correctly addressed")
                        .build(), props.helpfulnessMinLevel())
                .yesNo(PLAUSIBLE, YesNoQuestion.builder()
                        .text("Are all the numeric values in `assistant_answer` physically plausible for their units?")
                        .whenYes("Every value is within a range that can actually occur")
                        .whenNo("At least one value is impossible, such as a temperature below absolute zero "
                                + "or far outside anything ever recorded on Earth")
                        .build(), props.plausibleMinProbability())
                .build();
    }
}
