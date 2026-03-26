### Java 25 Spring Boot 4.0.4 MCP Testing

Spring AI MCP Server with `@McpTool` annotation exposing an exchange rate tool via the Streamable HTTP protocol.
Includes unit tests and integration tests using `McpSyncClient`.

### Stack
* Java 25
* Spring Boot 4.0.4
* Spring AI 1.1.2
* MCP SDK 0.17.0 (Streamable HTTP)
* Frankfurter API (exchange rates)

### How to build
```bash
mvn clean install
```

### How to run
```bash
./run.sh
```

### How to test (curl)
```bash
./test.sh
```

### test.sh output
```json
=== INITIALIZE ===
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "completions": {},
      "logging": {},
      "prompts": {
        "listChanged": true
      },
      "resources": {
        "subscribe": false,
        "listChanged": true
      },
      "tools": {
        "listChanged": true
      }
    },
    "serverInfo": {
      "name": "mcp-server",
      "version": "1.0.0"
    }
  }
}

=== LIST TOOLS ===
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "getExchangeRate",
        "title": "getExchangeRate",
        "description": "Get latest exchange rates for a base currency",
        "inputSchema": {
          "type": "object",
          "properties": {
            "base": {
              "type": "string",
              "description": "Base currency code, e.g. GBP, USD"
            }
          },
          "required": [
            "base"
          ]
        }
      }
    ]
  }
}

=== CALL TOOL (USD) ===
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"amount\":1.0,\"base\":\"USD\",\"date\":\"2026-03-25\",\"rates\":{\"AUD\":1.4361,\"BRL\":5.233,\"CAD\":1.3783,\"CHF\":0.7896,\"CNY\":6.8995,...}}"
      }
    ],
    "isError": false
  }
}

=== CALL TOOL (EUR) ===
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"amount\":1.0,\"base\":\"EUR\",\"date\":\"2026-03-25\",\"rates\":{\"AUD\":1.6647,\"BRL\":6.0661,\"CAD\":1.5977,\"CHF\":0.9153,\"CNY\":7.9979,...}}"
      }
    ],
    "isError": false
  }
}
```

### mvn test output
```
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.065 s -- in com.github.diegopacheco.mcptesting.ExchangeRateMcpToolUnitTest
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.967 s -- in com.github.diegopacheco.mcptesting.ExchangeRateMcpToolIntegrationTest
[INFO]
[INFO] Results:
[INFO]
[INFO] Tests run: 3, Failures: 0, Errors: 0, Skipped: 0
[INFO]
[INFO] BUILD SUCCESS
```
