package com.github.diegopacheco.sandboxspring.service;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

@Service
public class SubagentService {

    private final ChatClient orchestratorClient;

    public SubagentService(@Qualifier("orchestratorClient") ChatClient orchestratorClient) {
        this.orchestratorClient = orchestratorClient;
    }

    public String orchestrate(String task, String data) {
        String prompt = """
                Task: %s

                Data: %s
                """.formatted(task, data);
        return orchestratorClient.prompt()
                .user(prompt)
                .call()
                .content();
    }
}
