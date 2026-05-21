package bp

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.springframework.boot.SpringApplication
import org.springframework.context.ConfigurableApplicationContext

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}

class AppTests extends AnyFunSuite with BeforeAndAfterAll {

  private var ctx: ConfigurableApplicationContext = scala.compiletime.uninitialized
  private var port: Int = 0

  override def beforeAll(): Unit = {
    ctx = SpringApplication.run(classOf[App], "--server.port=0")
    port = ctx.getEnvironment.getProperty("local.server.port").toInt
  }

  override def afterAll(): Unit = {
    if (ctx != null) ctx.close()
  }

  private def get(path: String): HttpResponse[String] = {
    val c = HttpClient.newHttpClient()
    val r = HttpRequest.newBuilder(URI.create(s"http://127.0.0.1:$port$path")).GET().build()
    c.send(r, HttpResponse.BodyHandlers.ofString())
  }

  test("test_health_endpoint_returns_200") {
    assert(get("/actuator/health").statusCode() == 200)
  }

  test("test_health_payload_shape") {
    assert(get("/actuator/health").body().contains("\"status\":\"UP\""))
  }

  test("test_root_or_app_specific_smoke") {
    val r = get("/")
    assert(r.statusCode() == 200)
    assert(r.body() == "java25-scala3-sbt-sb4")
  }
}
