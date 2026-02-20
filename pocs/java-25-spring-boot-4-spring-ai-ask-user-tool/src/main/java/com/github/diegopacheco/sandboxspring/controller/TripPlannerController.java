package com.github.diegopacheco.sandboxspring.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/plan")
public class TripPlannerController {

    private final ChatClient chatClient;

    public TripPlannerController(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    @PostMapping
    public String plan(@RequestBody PlanRequest request) {
        return chatClient.prompt()
                .user(request.message())
                .call()
                .content();
    }

    record PlanRequest(String message) {}
}
