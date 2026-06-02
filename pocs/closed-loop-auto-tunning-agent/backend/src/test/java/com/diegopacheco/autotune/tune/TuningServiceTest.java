package com.diegopacheco.autotune.tune;

import com.diegopacheco.autotune.pattern.BreakerManager;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.json.JsonMapper;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class TuningServiceTest {

    @Test
    void rawProposalIsParsedThenClampedBeforeReturn() {
        OpenAiClient openai = mock(OpenAiClient.class);
        when(openai.model()).thenReturn("test-model");
        when(openai.complete(any(), any())).thenReturn("""
                {
                  "failureRateThreshold": 95,
                  "slowCallRateThreshold": 100,
                  "slowCallDurationThresholdMs": 50,
                  "slidingWindowType": "COUNT",
                  "slidingWindowSize": 5,
                  "minimumNumberOfCalls": 2,
                  "waitDurationInOpenStateSeconds": 0,
                  "permittedNumberOfCallsInHalfOpenState": 1,
                  "rationale": "lower everything"
                }
                """);

        TuningService service = new TuningService(openai, new BreakerManager(), JsonMapper.builder().build());

        TuneResult result = service.tune(null);

        assertThat(result.proposed().failureRateThreshold()).isEqualTo(95.0);
        assertThat(result.clamped().failureRateThreshold()).isEqualTo(70.0);
        assertThat(result.clamped().waitDurationInOpenStateSeconds()).isEqualTo(5L);
        assertThat(result.clamped().slowCallDurationThresholdMs()).isEqualTo(200L);
        assertThat(result.clamped().slidingWindowSize()).isEqualTo(10);
        assertThat(result.rationale()).isEqualTo("lower everything");
        assertThat(result.model()).isEqualTo("test-model");
        assertThat(result.clamps()).anyMatch(f -> f.field().equals("failureRateThreshold") && f.wasClamped());
    }
}
