package com.diegopacheco.autotune;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PatternEndpointIT {

    @LocalServerPort
    int port;

    private final HttpClient http = HttpClient.newHttpClient();

    @Test
    void circuitBreakerOpensAndShortCircuits() throws Exception {
        scenario(1.0, 0, 0);
        post("/api/cb/reset", "");

        List<String> outcomes = new ArrayList<>();
        for (int i = 0; i < 25; i++) {
            outcomes.add(outcomeOf(post("/api/cb/call", "")));
        }
        assertThat(outcomes).contains("SHORT_CIRCUITED");
    }

    @Test
    void rateLimiterRejectsExcessCalls() throws Exception {
        scenario(0.0, 0, 0);

        List<String> outcomes = new ArrayList<>();
        for (int i = 0; i < 15; i++) {
            outcomes.add(outcomeOf(post("/api/ratelimiter/call", "")));
        }
        assertThat(outcomes).contains("RATE_LIMITED");
    }

    @Test
    void bulkheadRejectsOverConcurrency() throws Exception {
        scenario(0.0, 200, 0);

        ExecutorService pool = Executors.newFixedThreadPool(12);
        List<Callable<String>> tasks = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            tasks.add(() -> outcomeOf(post("/api/bulkhead/call", "")));
        }
        List<Future<String>> futures = pool.invokeAll(tasks);
        List<String> outcomes = new ArrayList<>();
        for (Future<String> f : futures) {
            outcomes.add(f.get());
        }
        pool.shutdown();
        assertThat(outcomes).contains("REJECTED");
    }

    @Test
    void retryReturnsFailureWhenDownstreamAlwaysFails() throws Exception {
        scenario(1.0, 0, 0);
        assertThat(outcomeOf(post("/api/retry/call", ""))).isEqualTo("FAILURE");
    }

    private void scenario(double failRate, long latencyMs, long jitterMs) throws Exception {
        post("/api/sim/scenario",
                "{\"failRate\":" + failRate + ",\"latencyMs\":" + latencyMs + ",\"jitterMs\":" + jitterMs + "}");
    }

    private String post(String path, String body) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:" + port + path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
        return http.send(request, HttpResponse.BodyHandlers.ofString()).body();
    }

    private String outcomeOf(String json) {
        int i = json.indexOf("\"outcome\":\"");
        if (i < 0) {
            return "";
        }
        int start = i + "\"outcome\":\"".length();
        int end = json.indexOf('"', start);
        return json.substring(start, end);
    }
}
