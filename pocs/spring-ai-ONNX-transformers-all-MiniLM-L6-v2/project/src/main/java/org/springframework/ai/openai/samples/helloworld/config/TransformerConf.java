package org.springframework.ai.openai.samples.helloworld.config;

import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.transformers.TransformersEmbeddingClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class TransformerConf {

    @Bean("transformersEmbeddingClient")
    public EmbeddingClient embeddingClient() throws Exception {
        TransformersEmbeddingClient embeddingClient = new TransformersEmbeddingClient();
        embeddingClient.setTokenizerResource("classpath:/onnx/all-MiniLM-L6-v2/tokenizer.json");
        embeddingClient.setModelResource("classpath:/onnx/all-MiniLM-L6-v2/model.onnx");
        embeddingClient.setModelOutputName("token_embeddings");
        embeddingClient.afterPropertiesSet();
        return embeddingClient;
    }

}
