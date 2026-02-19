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

    @GetMapping("/skills")
    public String listSkills() {
        return chatClient.prompt()
                .user("List all the agent skills you have available and what each one does.")
                .call()
                .content();
    }
}
