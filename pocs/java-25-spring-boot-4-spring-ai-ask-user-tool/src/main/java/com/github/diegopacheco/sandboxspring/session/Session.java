package com.github.diegopacheco.sandboxspring.session;

import org.springaicommunity.agent.tools.AskUserQuestionTool;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

public class Session {

    public final String id;
    public volatile List<AskUserQuestionTool.Question> questions;
    public volatile CompletableFuture<Map<String, String>> answers;
    public final CompletableFuture<String> result = new CompletableFuture<>();
    public volatile CompletableFuture<Void> ready = new CompletableFuture<>();

    public Session(String id) {
        this.id = id;
    }

    public boolean isDone() {
        return result.isDone();
    }
}
