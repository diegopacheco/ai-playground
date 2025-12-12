package com.github.diegopacheco.embabel.agent;

import com.embabel.common.ai.model.Llm;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class EmbabelModelsConfiguration {
    @Bean
    public Llm defaultLlm(ChatModel chatModel, @Value("${embabel.models.default-llm:gpt-4o-mini}") String modelName) {
        return new Llm(modelName, "OpenAI", chatModel);
    }
}
