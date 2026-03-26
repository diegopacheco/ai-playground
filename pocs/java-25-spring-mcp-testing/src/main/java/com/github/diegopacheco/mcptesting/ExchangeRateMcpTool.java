package com.github.diegopacheco.mcptesting;

import org.springaicommunity.mcp.annotation.McpTool;
import org.springaicommunity.mcp.annotation.McpToolParam;
import org.springframework.stereotype.Component;

@Component
public class ExchangeRateMcpTool {
    private final ExchangeRateService exchangeRateService;

    public ExchangeRateMcpTool(ExchangeRateService exchangeRateService) {
        this.exchangeRateService = exchangeRateService;
    }

    @McpTool(description = "Get latest exchange rates for a base currency")
    public ExchangeRateResponse getExchangeRate(
        @McpToolParam(description = "Base currency code, e.g. GBP, USD", required = true) String base) {
        return exchangeRateService.getLatestExchangeRate(base);
    }
}
