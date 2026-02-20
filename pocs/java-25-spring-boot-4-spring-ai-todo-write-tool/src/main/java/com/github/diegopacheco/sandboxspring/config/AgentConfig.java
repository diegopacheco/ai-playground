package com.github.diegopacheco.sandboxspring.config;

import com.github.diegopacheco.sandboxspring.event.TodoUpdateEvent;
import org.springaicommunity.agent.tools.TodoWriteTool;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.ToolCallAdvisor;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AgentConfig {

    @Bean
    public ChatClient chatClient(ChatClient.Builder builder, ApplicationEventPublisher eventPublisher) {
        TodoWriteTool todoWriteTool = TodoWriteTool.builder()
                .todoEventHandler(event -> eventPublisher.publishEvent(
                        new TodoUpdateEvent(this, event.todos())))
                .build();

        return builder
                .defaultSystem("""
                        You are a helpful assistant that can manage and execute complex multi-step tasks.
                        When given a task with 3 or more distinct steps, use the TodoWrite tool to plan
                        and track your work. Mark tasks as in_progress when starting them and completed
                        when done. Only one task can be in_progress at a time.
                        """)
                .defaultTools(todoWriteTool)
                .defaultAdvisors(
                        ToolCallAdvisor.builder().conversationHistoryEnabled(false).build(),
                        MessageChatMemoryAdvisor.builder(MessageWindowChatMemory.builder().build()).build())
                .build();
    }
}
