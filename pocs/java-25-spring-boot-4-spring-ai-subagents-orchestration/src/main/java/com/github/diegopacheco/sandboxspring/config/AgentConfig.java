package com.github.diegopacheco.sandboxspring.config;

import java.util.List;

import org.springaicommunity.agent.tools.task.TaskTool;
import org.springaicommunity.agent.tools.task.claude.ClaudeSubagentReferences;
import org.springaicommunity.agent.tools.task.claude.ClaudeSubagentType;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;

@Configuration
public class AgentConfig {

    @Value("${agent.tasks.paths}")
    private List<Resource> agentPaths;

    @Bean
    public ChatClient orchestratorClient(ChatClient.Builder chatClientBuilder) {
        ToolCallback taskTool = TaskTool.builder()
                .subagentReferences(ClaudeSubagentReferences.fromResources(agentPaths))
                .subagentTypes(ClaudeSubagentType.builder()
                        .chatClientBuilder("default", chatClientBuilder.clone())
                        .build())
                .build();

        return chatClientBuilder.clone()
                .defaultSystem("""
                        You are a task orchestrator with access to specialized sub-agents via the Task tool.

                        Available agents:
                        - architect: Use for complex analysis, design decisions, and producing structured technical blueprints.
                        - builder: Use to generate final implementation, code, or polished output from a blueprint.

                        Guidelines:
                        - For complex tasks: use architect first to produce a blueprint, then builder to implement it.
                        - For simple tasks: use builder directly.
                        - Always return the final response to the user.
                        """)
                .defaultToolCallbacks(taskTool)
                .build();
    }
}
