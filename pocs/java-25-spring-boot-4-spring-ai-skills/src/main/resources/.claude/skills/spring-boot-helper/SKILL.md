---
name: spring-boot-helper
description: Expert knowledge of Spring Boot 4 and Spring AI 2.0 including auto-configuration,
  ChatClient, advisors, tool calling, agent patterns, and Spring Framework 7 features.
  Use when the user asks about Spring Boot, Spring AI, dependency injection, REST APIs,
  agent configuration, or Spring ecosystem questions.
allowed-tools: Read, Grep
---

# Spring Boot 4 & Spring AI 2.0 Helper Skill

## Overview
Provides expert guidance on Spring Boot 4, Spring Framework 7, and Spring AI 2.0.

## Spring Boot 4 Key Changes
- Requires Java 17+ (Java 25 fully supported)
- Built on Spring Framework 7
- Jakarta EE 11 namespace (`jakarta.*` everywhere)
- Virtual threads auto-configured via `spring.threads.virtual.enabled=true`

## Spring AI 2.0 ChatClient
```java
ChatClient chatClient = chatClientBuilder
    .defaultSystem("You are a helpful assistant")
    .defaultTools(myTool)
    .build();

String response = chatClient.prompt()
    .user("Hello!")
    .call()
    .content();
```

## Tool Calling
```java
@Bean
@Description("Get the current weather for a city")
public Function<WeatherRequest, WeatherResponse> weatherFunction() {
    return request -> new WeatherResponse(72.0, "sunny");
}
```

## Advisors
- `MessageChatMemoryAdvisor` - conversation memory
- `ToolCallAdvisor` - tool execution
- Custom advisors implement `CallAroundAdvisor`

## Agent Patterns
- **Skill-based agents**: Use SkillsTool with Markdown skill files
- **Sub-agents**: Delegate tasks via TaskTool to specialized models
- **Memory**: MessageWindowChatMemory for conversation history

## Auto-configuration
- Set `OPENAI_API_KEY` env var
- Property: `spring.ai.openai.chat.options.model=gpt-4o`

## Virtual Threads with Spring Boot 4
```properties
spring.threads.virtual.enabled=true
```
Automatically uses virtual threads for Tomcat and @Async tasks.
