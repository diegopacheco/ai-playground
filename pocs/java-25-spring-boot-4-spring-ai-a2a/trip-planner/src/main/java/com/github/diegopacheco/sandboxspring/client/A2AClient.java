package com.github.diegopacheco.sandboxspring.client;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.List;
import java.util.UUID;

@Component
public class A2AClient {

    private final RestClient restClient = RestClient.create();

    public String sendMessage(String agentUrl, String message) {
        var request = new JsonRpcRequest(
                "2.0",
                "message/send",
                UUID.randomUUID().toString(),
                new Params(new Message("message", UUID.randomUUID().toString(), "user", List.of(new Part("text", message))))
        );

        var response = restClient.post()
                .uri(agentUrl)
                .contentType(MediaType.APPLICATION_JSON)
                .body(request)
                .retrieve()
                .body(JsonRpcResponse.class);

        if (response == null || response.result() == null) {
            return "No response received from agent at " + agentUrl;
        }

        if (response.result().artifacts() == null || response.result().artifacts().isEmpty()) {
            return "Agent returned no artifacts";
        }

        return response.result().artifacts().stream()
                .flatMap(a -> a.parts().stream())
                .filter(p -> "text".equals(p.kind()))
                .map(Part::text)
                .findFirst()
                .orElse("No text response from agent");
    }

    record JsonRpcRequest(String jsonrpc, String method, String id, Params params) {}

    record Params(Message message) {}

    record Message(String kind, String messageId, String role, List<Part> parts) {}

    record Part(String kind, String text) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record JsonRpcResponse(String jsonrpc, String id, TaskResult result) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record TaskResult(String id, TaskStatus status, List<Artifact> artifacts) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record TaskStatus(String state) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    record Artifact(List<Part> parts) {}
}
