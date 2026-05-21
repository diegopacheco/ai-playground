package bp;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.ResponseEntity;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class AppTests {

    @Autowired
    private TestRestTemplate rest;

    @Test
    void test_health_endpoint_returns_200() {
        ResponseEntity<String> r = rest.getForEntity("/actuator/health", String.class);
        assertEquals(200, r.getStatusCodeValue());
    }

    @Test
    void test_health_payload_shape() {
        ResponseEntity<String> r = rest.getForEntity("/actuator/health", String.class);
        assertTrue(r.getBody() != null && r.getBody().contains("\"status\":\"UP\""));
    }

    @Test
    void test_root_or_app_specific_smoke() {
        ResponseEntity<String> r = rest.getForEntity("/", String.class);
        assertEquals(200, r.getStatusCodeValue());
        assertEquals("java8-mvn-sb2", r.getBody());
    }
}
