package com.diegopacheco.autotune.tune;

import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

@Component
public class OpenAiClient {

    public record ChatResponse(List<Choice> choices) {}

    public record Choice(Message message) {}

    public record Message(String content) {}

    private final ObjectMapper mapper;
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
    private final String apiKey = System.getenv("OPENAI_API_KEY");
    private final String model = System.getenv().getOrDefault("OPENAI_MODEL", "gpt-4o");

    public OpenAiClient(ObjectMapper mapper) {
        this.mapper = mapper;
    }

    public boolean configured() {
        return apiKey != null && !apiKey.isBlank();
    }

    public String model() {
        return model;
    }

    public String complete(String system, String user) {
        String body = mapper.writeValueAsString(Map.of(
                "model", model,
                "temperature", 0.2,
                "response_format", Map.of("type", "json_object"),
                "messages", List.of(
                        Map.of("role", "system", "content", system),
                        Map.of("role", "user", "content", user))));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.openai.com/v1/chat/completions"))
                .timeout(Duration.ofSeconds(60))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response;
        try {
            response = http.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (Exception e) {
            throw new RuntimeException("OpenAI request failed: " + e.getMessage(), e);
        }
        if (response.statusCode() / 100 != 2) {
            throw new RuntimeException("OpenAI returned " + response.statusCode() + ": " + response.body());
        }
        ChatResponse parsed = mapper.readValue(response.body(), ChatResponse.class);
        if (parsed.choices() == null || parsed.choices().isEmpty()) {
            throw new RuntimeException("OpenAI returned no choices");
        }
        return parsed.choices().get(0).message().content();
    }
}
