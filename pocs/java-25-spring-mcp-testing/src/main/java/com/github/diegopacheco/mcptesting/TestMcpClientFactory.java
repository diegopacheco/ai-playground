package com.github.diegopacheco.mcptesting;

import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class TestMcpClientFactory {
    private final String protocol;

    public TestMcpClientFactory(@Value("${spring.ai.mcp.server.protocol:streamable}") String protocol) {
        this.protocol = protocol;
    }

    public McpSyncClient create(String baseUrl) {
        return McpClient.sync(HttpClientStreamableHttpTransport.builder(baseUrl)
          .endpoint("/mcp")
          .build()
        ).build();
    }
}
