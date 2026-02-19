package com.github.diegopacheco.sandboxspring.config;

import com.github.diegopacheco.sandboxspring.tools.HotelTools;
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
public class HotelAgentConfig {

    @Bean
    public AgentCard agentCard(@Value("${server.port:10002}") int port) {
        return AgentCard.builder()
                .name("Hotel Agent")
                .description("Provides hotel search, recommendations and availability for travel destinations")
                .url("http://localhost:" + port + "/a2a/")
                .version("1.0.0")
                .capabilities(AgentCapabilities.builder().streaming(false).build())
                .defaultInputModes(List.of("text"))
                .defaultOutputModes(List.of("text"))
                .skills(List.of(
                        AgentSkill.builder()
                                .id("hotel_search")
                                .name("Hotel Search")
                                .description("Search for hotels by destination and budget tier")
                                .tags(List.of("hotel", "search", "accommodation"))
                                .build(),
                        AgentSkill.builder()
                                .id("hotel_availability")
                                .name("Hotel Availability")
                                .description("Check hotel availability and pricing for specific dates")
                                .tags(List.of("hotel", "availability", "booking"))
                                .build()
                ))
                .build();
    }

    @Bean
    public AgentExecutor agentExecutor(ChatClient.Builder builder, HotelTools tools) {
        ChatClient chatClient = builder.clone()
                .defaultSystem("You are a hotel and accommodation expert. Use available tools to help travelers find the best hotels for their needs.")
                .defaultTools(tools)
                .build();
        return new DefaultAgentExecutor(chatClient, (chat, ctx) -> {
            String msg = DefaultAgentExecutor.extractTextFromMessage(ctx.getMessage());
            return chat.prompt().user(msg).call().content();
        });
    }
}
