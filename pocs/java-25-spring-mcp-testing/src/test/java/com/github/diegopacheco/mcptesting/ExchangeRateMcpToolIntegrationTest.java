package com.github.diegopacheco.mcptesting;

import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.spec.McpSchema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import java.util.Map;
import java.util.Objects;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class ExchangeRateMcpToolIntegrationTest {
    @LocalServerPort
    private int port;

    @Autowired
    private TestMcpClientFactory testMcpClientFactory;

    @MockitoBean
    private ExchangeRateService exchangeRateService;

    private McpSyncClient client;

    @BeforeEach
    void setUp() {
        client = testMcpClientFactory.create("http://localhost:" + port);
        client.initialize();
    }

    @AfterEach
    void cleanUp() {
        client.closeGracefully();
    }

    @Test
    void whenMcpClientListTools_thenTheToolIsRegistered() {
        boolean registered = client.listTools().tools().stream()
          .anyMatch(tool -> Objects.equals(tool.name(), "getExchangeRate"));
        assertThat(registered).isTrue();
    }

    @Test
    void whenMcpClientCallTool_thenTheToolReturnsMockedResponse() {
        when(exchangeRateService.getLatestExchangeRate("GBP")).thenReturn(
          new ExchangeRateResponse(1.0, "GBP", "2026-03-08", Map.of("USD", 1.27))
        );

        McpSchema.Tool exchangeRateTool = client.listTools().tools().stream()
          .filter(tool -> "getExchangeRate".equals(tool.name()))
          .findFirst()
          .orElseThrow();

        String argumentName = exchangeRateTool.inputSchema().properties().keySet().stream()
          .findFirst()
          .orElseThrow();

        McpSchema.CallToolResult result = client.callTool(
          new McpSchema.CallToolRequest("getExchangeRate", Map.of(argumentName, "GBP"))
        );

        assertThat(result).isNotNull();
        assertThat(result.isError()).isFalse();
        assertTrue(result.toString().contains("GBP"));
    }
}
