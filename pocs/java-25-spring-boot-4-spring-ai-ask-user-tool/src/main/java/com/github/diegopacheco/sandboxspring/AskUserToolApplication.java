package com.github.diegopacheco.sandboxspring;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class AskUserToolApplication implements CommandLineRunner {

    private final ChatClient chatClient;

    public AskUserToolApplication(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    public static void main(String[] args) {
        SpringApplication.run(AskUserToolApplication.class, args);
    }

    @Override
    public void run(String... args) {
        System.out.println("\n=== Spring AI Ask User Question Tool ===\n");

        String response = chatClient.prompt()
                .user("I want to plan a trip. Please help me plan the perfect vacation.")
                .call()
                .content();

        System.out.println("\n=== Your Personalized Trip Plan ===\n");
        System.out.println(response);
    }
}
