package com.github.diegopacheco.qdrant.amazon;

import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
class OllamaClient {

    private final RestClient http;
    private final String embedModel;
    private final String chatModel;

    record EmbedResponse(List<float[]> embeddings) {}

    record ChatResponse(Message message) {}

    record Message(String role, String content) {}

    OllamaClient(@Value("${ollama.url}") String url,
                 @Value("${ollama.embed-model}") String embedModel,
                 @Value("${ollama.chat-model}") String chatModel) {
        this.http = RestClient.create(url);
        this.embedModel = embedModel;
        this.chatModel = chatModel;
    }

    List<float[]> embed(List<String> texts) {
        return http.post()
                .uri("/api/embed")
                .body(Map.of("model", embedModel, "input", texts))
                .retrieve()
                .body(EmbedResponse.class)
                .embeddings();
    }

    String chat(String prompt) {
        return http.post()
                .uri("/api/chat")
                .body(Map.of("model", chatModel, "stream", false, "messages", List.of(new Message("user", prompt))))
                .retrieve()
                .body(ChatResponse.class)
                .message()
                .content();
    }
}
