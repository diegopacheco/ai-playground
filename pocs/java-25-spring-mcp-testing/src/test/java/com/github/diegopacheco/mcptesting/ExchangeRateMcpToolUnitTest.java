package com.github.diegopacheco.mcptesting;

import org.junit.jupiter.api.Test;
import java.util.Map;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class ExchangeRateMcpToolUnitTest {
    @Test
    void whenBaseIsNotBlank_thenGetExchangeRateShouldReturnResponse() {
        ExchangeRateService exchangeRateService = mock(ExchangeRateService.class);
        ExchangeRateResponse expected = new ExchangeRateResponse(1.0, "GBP", "2026-03-08",
          Map.of("USD", 1.27, "EUR", 1.17));
        when(exchangeRateService.getLatestExchangeRate("gbp")).thenReturn(expected);

        ExchangeRateMcpTool tool = new ExchangeRateMcpTool(exchangeRateService);
        ExchangeRateResponse actual = tool.getExchangeRate("gbp");

        assertThat(actual).isEqualTo(expected);
        verify(exchangeRateService).getLatestExchangeRate("gbp");
    }
}
