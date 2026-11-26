package com.github.diegopacheco.decisions.triage;

import java.time.Clock;
import java.util.Map;

import com.github.diegopacheco.decisions.config.DecisionsProperties;
import com.github.diegopacheco.decisions.logging.AppLogger;
import com.github.diegopacheco.decisions.web.Stopwatch;

import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.answer.ScoreAnswer;
import org.springframework.ai.decision.answer.YesNoAnswer;
import org.springframework.stereotype.Service;

@Service
public class TriageService {

    private static final AppLogger log = AppLogger.of(TriageService.class);

    private final DecisionModel decisionModel;
    private final DecisionsProperties props;
    private final Clock clock;

    public TriageService(DecisionModel decisionModel, DecisionsProperties props, Clock clock) {
        this.decisionModel = decisionModel;
        this.props = props;
        this.clock = clock;
    }

    public TriageResult triage(String message) {
        Stopwatch.Timed<DecisionResponse> timed = Stopwatch.time(clock,
                () -> decisionModel.call(TriageQuestions.prompt(message)));
        DecisionResponse response = timed.value();
        YesNoAnswer defect = response.getYesNo(TriageQuestions.DEFECT);
        ChoiceAnswer topic = response.getChoice(TriageQuestions.TOPIC);
        ScoreAnswer satisfaction = response.getScore(TriageQuestions.SATISFACTION);
        Route route = Route.of(defect.getProbability(), props.autoThreshold(), props.reviewThreshold());
        log.info("triage decided", Map.of("route", route, "defect", defect.getProbability(), "ms", timed.millis()));
        return new TriageResult(response.getMetadata().getModel(), timed.millis(),
                new TriageResult.Defect(defect.isYes(), defect.getProbability()),
                new TriageResult.Topic(topic.getValue(), topic.getProbabilities()),
                new TriageResult.Satisfaction(TriageQuestions.SATISFACTION_LEVELS.get(satisfaction.getNearestLevel()),
                        satisfaction.getNearestLevel(), satisfaction.getValue(), TriageQuestions.SATISFACTION_LEVELS,
                        satisfaction.getProbabilities()),
                route, new TriageResult.Thresholds(props.autoThreshold(), props.reviewThreshold()));
    }
}
