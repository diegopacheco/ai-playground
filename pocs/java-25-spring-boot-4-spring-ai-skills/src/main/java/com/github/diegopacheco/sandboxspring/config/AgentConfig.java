package com.github.diegopacheco.sandboxspring.config;

import org.springaicommunity.agent.tools.SkillsTool;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;

@Configuration
public class AgentConfig {

    @Bean
    public ChatClient chatClient(ChatClient.Builder builder, ResourceLoader resourceLoader) {
        Resource skillsResource = resourceLoader.getResource("classpath:.claude/skills");
        ToolCallback skillsTool = SkillsTool.builder()
                .addSkillsResource(skillsResource)
                .build();
        return builder
                .defaultSystem("""
                        You are an expert Java and Spring AI assistant with deep knowledge of
                        Java 25 and Spring Boot 4. You have access to specialized skills that
                        provide detailed guidance. Use them when relevant to answer questions.
                        """)
                .defaultToolCallbacks(skillsTool)
                .build();
    }
}
