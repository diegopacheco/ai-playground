package org.springframework.ai.openai.samples.helloworld.simple;

import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.transformers.TransformersEmbeddingClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

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

	@GetMapping("/ai")
	public List<List<Double>> getResult() throws Exception {
		TransformersEmbeddingClient embeddingClient = new TransformersEmbeddingClient();

		// (optional) defaults to classpath:/onnx/all-MiniLM-L6-v2/tokenizer.json
		embeddingClient.setTokenizerResource("classpath:/onnx/all-MiniLM-L6-v2/tokenizer.json");

		// (optional) defaults to classpath:/onnx/all-MiniLM-L6-v2/model.onnx
		embeddingClient.setModelResource("classpath:/onnx/all-MiniLM-L6-v2/model.onnx");

		embeddingClient.setModelOutputName("token_embeddings");

		// (optional) defaults to ${java.io.tmpdir}/spring-ai-onnx-model
		// Only the http/https resources are cached by default.
		embeddingClient.setResourceCacheDirectory("/tmp/onnx-zoo");

		// (optional) Set the tokenizer padding if you see an errors like:
		// "ai.onnxruntime.OrtException: Supplied array is ragged, ..."
		embeddingClient.setTokenizerOptions(Map.of("padding", "true"));

		embeddingClient.afterPropertiesSet();

		List<List<Double>> embeddings = embeddingClient.embed(List.of("Hello world", "World is big"));
		return embeddings;
	}
}
