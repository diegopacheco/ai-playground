package com.github.diegopacheco.sandboxspring.session;

import org.springaicommunity.agent.tools.AskUserQuestionTool;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class SessionQuestionHandler implements AskUserQuestionTool.QuestionHandler {

    private final Session session;

    public SessionQuestionHandler(Session session) {
        this.session = session;
    }

    @Override
    public Map<String, String> handle(List<AskUserQuestionTool.Question> questions) {
        session.questions = questions;
        session.answers = new CompletableFuture<>();
        CompletableFuture<Void> current = session.ready;
        session.ready = new CompletableFuture<>();
        current.complete(null);
        try {
            return session.answers.get(5, TimeUnit.MINUTES);
        } catch (Exception e) {
            throw new RuntimeException("Timeout waiting for user answer", e);
        }
    }
}
