package com.github.diegopacheco.sandboxspring.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/agent")
public class AgentController {

    private final ChatClient chatClient;

    public AgentController(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    @PostMapping("/ask")
    public String ask(@RequestBody String prompt) {
        return chatClient.prompt()
                .user(prompt)
                .call()
                .content();
    }

    @GetMapping("/demo")
    public String demo() {
        return chatClient.prompt()
                .user("Find the top 5 Java design patterns, describe each one briefly, " +
                      "give a real-world use case for each, and provide a summary. " +
                      "Use TodoWrite to organize your tasks.")
                .call()
                .content();
    }
}
