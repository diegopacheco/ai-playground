package bp

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class AppTests {

    @LocalServerPort
    var port: Int = 0

    private fun get(path: String): HttpResponse<String> {
        val c = HttpClient.newHttpClient()
        val r = HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port$path")).GET().build()
        return c.send(r, HttpResponse.BodyHandlers.ofString())
    }

    @Test
    fun test_health_endpoint_returns_200() {
        assertEquals(200, get("/actuator/health").statusCode())
    }

    @Test
    fun test_health_payload_shape() {
        assertTrue(get("/actuator/health").body().contains("\"status\":\"UP\""))
    }

    @Test
    fun test_root_or_app_specific_smoke() {
        val r = get("/")
        assertEquals(200, r.statusCode())
        assertEquals("java25-kotlin-mvn-sb4", r.body())
    }
}
