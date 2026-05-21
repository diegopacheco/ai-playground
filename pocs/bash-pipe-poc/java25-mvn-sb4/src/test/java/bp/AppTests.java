package bp;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class AppTests {

    @LocalServerPort
    int port;

    private HttpResponse<String> get(String path) throws Exception {
        HttpClient c = HttpClient.newHttpClient();
        HttpRequest r = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + path)).GET().build();
        return c.send(r, HttpResponse.BodyHandlers.ofString());
    }

    @Test
    void test_health_endpoint_returns_200() throws Exception {
        HttpResponse<String> r = get("/actuator/health");
        assertEquals(200, r.statusCode());
    }

    @Test
    void test_health_payload_shape() throws Exception {
        HttpResponse<String> r = get("/actuator/health");
        assertTrue(r.body().contains("\"status\":\"UP\""));
    }

    @Test
    void test_root_or_app_specific_smoke() throws Exception {
        HttpResponse<String> r = get("/");
        assertEquals(200, r.statusCode());
        assertEquals("java25-mvn-sb4", r.body());
    }
}
