package com.github.diegopacheco.sandboxspring.config;

import com.github.diegopacheco.sandboxspring.tools.WeatherTools;
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
public class WeatherAgentConfig {

    @Bean
    public AgentCard agentCard(@Value("${server.port:10001}") int port) {
        return new AgentCard.Builder()
                .name("Weather Agent")
                .description("Provides current weather conditions and forecasts for any location worldwide")
                .url("http://localhost:" + port + "/a2a/")
                .version("1.0.0")
                .capabilities(new AgentCapabilities.Builder().streaming(false).build())
                .defaultInputModes(List.of("text"))
                .defaultOutputModes(List.of("text"))
                .skills(List.of(
                        new AgentSkill.Builder()
                                .id("current_weather")
                                .name("Current Weather")
                                .description("Get current weather conditions for any location")
                                .tags(List.of("weather", "current", "temperature"))
                                .build(),
                        new AgentSkill.Builder()
                                .id("weather_forecast")
                                .name("Weather Forecast")
                                .description("Get multi-day weather forecast for any location")
                                .tags(List.of("weather", "forecast"))
                                .build()
                ))
                .build();
    }

    @Bean
    public AgentExecutor agentExecutor(ChatClient.Builder builder, WeatherTools tools) {
        ChatClient chatClient = builder.clone()
                .defaultSystem("You are a weather expert. Use available tools to provide detailed weather information for requested locations.")
                .defaultTools(tools)
                .build();
        return new DefaultAgentExecutor(chatClient, (chat, ctx) -> {
            String msg = DefaultAgentExecutor.extractTextFromMessage(ctx.getMessage());
            return chat.prompt().user(msg).call().content();
        });
    }
}
