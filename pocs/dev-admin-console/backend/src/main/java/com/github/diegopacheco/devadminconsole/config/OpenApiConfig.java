package com.github.diegopacheco.devadminconsole.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {
    @Bean
    public OpenAPI openApi() {
        return new OpenAPI().info(new Info()
                .title("Dev Admin Console")
                .version("1.0.0")
                .description("Read-only console for Cassandra, MySQL, Postgres, Redis, etcd, Kafka and Elasticsearch"));
    }
}
