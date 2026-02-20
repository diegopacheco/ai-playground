package com.github.diegopacheco.sandboxspring.handler;

import org.springaicommunity.agent.tools.AskUserQuestionTool;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public class SimulatedQuestionHandler implements AskUserQuestionTool.QuestionHandler {

    @Override
    public Map<String, String> handle(List<AskUserQuestionTool.Question> questions) {
        Map<String, String> answers = new LinkedHashMap<>();
        for (AskUserQuestionTool.Question q : questions) {
            System.out.println("[AskUserTool] Question: " + q.question());
            List<AskUserQuestionTool.Question.Option> options = q.options();
            options.forEach(o -> System.out.println("  - " + o.label() + ": " + o.description()));
            String answer = options.isEmpty() ? "Yes" : options.getFirst().label();
            System.out.println("[AskUserTool] Simulated answer: " + answer);
            answers.put(q.question(), answer);
        }
        return answers;
    }
}
