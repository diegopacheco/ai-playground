package com.github.diegopacheco.decisions.judge;

import java.time.Clock;
import java.util.List;
import java.util.Map;

import com.github.diegopacheco.decisions.config.DecisionsProperties;
import com.github.diegopacheco.decisions.logging.AppLogger;
import com.github.diegopacheco.decisions.web.Stopwatch;

import org.springaicommunity.decision.judge.DecisionFinding;
import org.springaicommunity.decision.judge.DecisionJudge;
import org.springaicommunity.decision.judge.DecisionVerdict;
import org.springframework.ai.decision.answer.ScoreAnswer;
import org.springframework.ai.decision.answer.YesNoAnswer;
import org.springframework.stereotype.Service;

@Service
public class JudgeService {

    private static final AppLogger log = AppLogger.of(JudgeService.class);

    private static final double SCORE_SUPPORT_NEEDED = 0.5;

    private final DecisionJudge judge;
    private final DecisionsProperties props;
    private final Clock clock;

    public JudgeService(DecisionJudge judge, DecisionsProperties props, Clock clock) {
        this.judge = judge;
        this.props = props;
        this.clock = clock;
    }

    public JudgeResult judge(String question, String answer) {
        Stopwatch.Timed<DecisionVerdict> timed = Stopwatch.time(clock, () -> judge.judge(question, answer));
        DecisionVerdict verdict = timed.value();
        List<JudgeResult.Finding> findings = verdict.findings().stream().map(this::finding).toList();
        log.info("judge decided", Map.of("passed", verdict.passed(), "ms", timed.millis()));
        return new JudgeResult(timed.millis(), verdict.passed(), findings,
                verdict.passed() ? "" : verdict.feedback());
    }

    private JudgeResult.Finding scoreFinding(DecisionFinding finding, String outcome, ScoreAnswer score) {
        int minLevel = (int) Math.ceil(props.helpfulnessMinLevel());
        List<Double> probabilities = score.getProbabilities();
        if (probabilities.isEmpty()) {
            return new JudgeResult.Finding(finding.name(), outcome, "expected level of 0 to 3", score.getValue(),
                    props.helpfulnessMinLevel(), finding.detail());
        }
        double support = probabilities.stream().skip(minLevel).mapToDouble(Double::doubleValue).sum();
        String measure = "probability of level %d or higher (expected level %.2f)".formatted(minLevel, score.getValue());
        return new JudgeResult.Finding(finding.name(), outcome, measure, support, SCORE_SUPPORT_NEEDED, finding.detail());
    }

    private JudgeResult.Finding finding(DecisionFinding finding) {
        String outcome = finding.outcome().name();
        return switch (finding.answer()) {
            case YesNoAnswer yesNo -> new JudgeResult.Finding(finding.name(), outcome, "probability of yes",
                    yesNo.getProbability(), props.plausibleMinProbability(), finding.detail());
            case ScoreAnswer score -> scoreFinding(finding, outcome, score);
            default -> throw new IllegalStateException("unexpected answer for " + finding.name());
        };
    }
}
