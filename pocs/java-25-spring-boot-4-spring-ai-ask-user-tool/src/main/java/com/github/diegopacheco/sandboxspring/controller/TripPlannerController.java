package com.github.diegopacheco.sandboxspring.controller;

import com.github.diegopacheco.sandboxspring.session.Session;
import com.github.diegopacheco.sandboxspring.session.SessionQuestionHandler;
import com.github.diegopacheco.sandboxspring.session.SessionStore;
import org.springaicommunity.agent.tools.AskUserQuestionTool;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

@RestController
@RequestMapping("/plan")
public class TripPlannerController {

    private static final String SYSTEM = """
            You are a travel advisor. Before planning a trip, always use the askUserQuestion tool
            to gather the traveler's preferences. Ask about destination type, budget range,
            trip duration, and travel style. Then provide a detailed, personalized trip plan
            based on their answers.
            """;

    private final ChatClient.Builder chatClientBuilder;
    private final SessionStore sessionStore;

    public TripPlannerController(ChatClient.Builder chatClientBuilder, SessionStore sessionStore) {
        this.chatClientBuilder = chatClientBuilder;
        this.sessionStore = sessionStore;
    }

    @PostMapping("/start")
    public StartResponse start(@RequestBody StartRequest request) throws Exception {
        Session session = sessionStore.create();
        SessionQuestionHandler handler = new SessionQuestionHandler(session);
        AskUserQuestionTool tool = AskUserQuestionTool.builder()
                .questionHandler(handler)
                .build();
        ChatClient client = chatClientBuilder.clone()
                .defaultSystem(SYSTEM)
                .defaultTools(tool)
                .build();

        CompletableFuture.runAsync(() -> {
            try {
                String r = client.prompt().user(request.message()).call().content();
                session.result.complete(r);
            } catch (Exception e) {
                session.result.completeExceptionally(e);
            } finally {
                session.ready.complete(null);
            }
        });

        session.ready.get(30, TimeUnit.SECONDS);
        return new StartResponse(session.id);
    }

    @GetMapping("/question/{id}")
    public QuestionResponse getQuestion(@PathVariable String id) throws Exception {
        Session session = sessionStore.get(id);
        if (session.isDone()) {
            String result = session.result.get();
            sessionStore.remove(id);
            return new QuestionResponse(id, "done", null, result);
        }
        List<QuestionDto> dtos = session.questions.stream()
                .map(q -> new QuestionDto(
                        q.question(), q.header(),
                        Boolean.TRUE.equals(q.multiSelect()),
                        q.options().stream().map(o -> new OptionDto(o.label(), o.description())).toList()
                ))
                .toList();
        return new QuestionResponse(id, "asking", dtos, null);
    }

    @PostMapping("/answer/{id}")
    public AnswerResponse submitAnswer(@PathVariable String id, @RequestBody AnswerRequest request) throws Exception {
        Session session = sessionStore.get(id);
        CompletableFuture<Void> nextReady = session.ready;
        session.answers.complete(request.answers());
        nextReady.get(30, TimeUnit.SECONDS);
        return new AnswerResponse(id, session.isDone() ? "done" : "asking");
    }

    record StartRequest(String message) {}
    record StartResponse(String id) {}
    record OptionDto(String label, String description) {}
    record QuestionDto(String question, String header, boolean multiSelect, List<OptionDto> options) {}
    record QuestionResponse(String id, String status, List<QuestionDto> questions, String result) {}
    record AnswerRequest(Map<String, String> answers) {}
    record AnswerResponse(String id, String status) {}
}
