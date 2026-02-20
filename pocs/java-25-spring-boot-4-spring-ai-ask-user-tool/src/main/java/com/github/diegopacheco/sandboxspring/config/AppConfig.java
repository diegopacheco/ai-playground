package com.github.diegopacheco.sandboxspring.config;

import com.github.diegopacheco.sandboxspring.handler.SimulatedQuestionHandler;
import org.springaicommunity.agent.tools.AskUserQuestionTool;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {

    @Bean
    public AskUserQuestionTool askUserQuestionTool(SimulatedQuestionHandler handler) {
        return AskUserQuestionTool.builder()
                .questionHandler(handler)
                .build();
    }

    @Bean
    public ChatClient chatClient(ChatClient.Builder builder, AskUserQuestionTool askUserQuestionTool) {
        return builder
                .defaultSystem("""
                        You are a travel advisor. Before planning a trip, always use the askUserQuestion tool
                        to gather the traveler's preferences. Ask about destination type, budget range,
                        trip duration, and travel style. Then provide a detailed, personalized trip plan
                        based on their answers.
                        """)
                .defaultTools(askUserQuestionTool)
                .build();
    }
}
