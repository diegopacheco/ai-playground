package org.springframework.ai.openai.samples.helloworld.simple;

import org.springframework.ai.transformers.TransformersEmbeddingClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class SimpleAiController {

    //private final ChatClient chatClient;

    //@Autowired
    //public SimpleAiController(ChatClient chatClient) {
    //this.chatClient = chatClient;
    //}

    //@GetMapping("/ai/simple")
    //public Map<String, String> completion(@RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
    //		return Map.of("generation", chatClient.call(message));
    //}

    @Autowired
    private TransformersEmbeddingClient embeddingClient;

    @GetMapping("/ai")
    public List<List<Double>> getResult() throws Exception {
        List<List<Double>> embeddings = embeddingClient.embed(List.of("Hello world", "World is big"));
        return embeddings;
    }
}
