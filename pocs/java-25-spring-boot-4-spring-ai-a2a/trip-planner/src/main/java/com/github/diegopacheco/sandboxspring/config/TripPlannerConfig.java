package com.github.diegopacheco.sandboxspring.config;

import com.github.diegopacheco.sandboxspring.tools.RemoteAgentTools;
import io.a2a.server.agentexecution.AgentExecutor;
import io.a2a.spec.AgentCapabilities;
import io.a2a.spec.AgentCard;
import io.a2a.spec.AgentSkill;
import org.springaicommunity.a2a.server.executor.DefaultAgentExecutor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class TripPlannerConfig {

    @Bean
    public AgentCard agentCard(@Value("${server.port:10000}") int port) {
        return new AgentCard.Builder()
                .name("Trip Planner Agent")
                .description("Orchestrates weather and hotel agents via A2A protocol to create comprehensive trip plans")
                .url("http://localhost:" + port + "/a2a/")
                .version("1.0.0")
                .capabilities(new AgentCapabilities.Builder().streaming(false).build())
                .defaultInputModes(List.of("text"))
                .defaultOutputModes(List.of("text"))
                .skills(List.of(
                        new AgentSkill.Builder()
                                .id("trip_planning")
                                .name("Trip Planning")
                                .description("Plan complete trips with weather forecasts and hotel recommendations via agent orchestration")
                                .tags(List.of("trip", "travel", "planning", "orchestration", "a2a"))
                                .build()
                ))
                .build();
    }

    @Bean
    public ChatClient tripPlannerChatClient(ChatClient.Builder builder, RemoteAgentTools tools) {
        return builder.clone()
                .defaultSystem("""
                        You are a comprehensive trip planning orchestrator with access to specialized A2A agents.
                        When planning a trip:
                        1. Get weather information for the destination using the weather agent
                        2. Search for hotels matching the traveler's budget using the hotel agent
                        3. Combine the information into a clear, actionable trip plan with specific recommendations
                        Always provide practical, specific advice based on data from the specialized agents.
                        """)
                .defaultTools(tools)
                .build();
    }

    @Bean
    public AgentExecutor agentExecutor(ChatClient tripPlannerChatClient) {
        return new DefaultAgentExecutor(tripPlannerChatClient, (chat, ctx) -> {
            String msg = DefaultAgentExecutor.extractTextFromMessage(ctx.getMessage());
            return chat.prompt().user(msg).call().content();
        });
    }
}
