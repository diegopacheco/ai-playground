package bp

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}

class AppTests extends AnyFunSuite with BeforeAndAfterAll {

  private val testPort = 8096

  override def beforeAll(): Unit = {
    bp.App.runOn(testPort)
    waitReady()
  }

  private def get(path: String): HttpResponse[String] = {
    val c = HttpClient.newHttpClient()
    val r = HttpRequest.newBuilder(URI.create(s"http://127.0.0.1:$testPort$path")).GET().build()
    c.send(r, HttpResponse.BodyHandlers.ofString())
  }

  private def waitReady(): Unit = {
    var i = 0
    var ok = false
    while (!ok && i < 50) {
      try { if (get("/health").statusCode() == 200) ok = true }
      catch { case _: Throwable => Thread.sleep(100) }
      i += 1
    }
    if (!ok) throw new RuntimeException(s"server not ready on $testPort")
  }

  test("test_health_endpoint_returns_200") {
    assert(get("/health").statusCode() == 200)
  }

  test("test_health_payload_shape") {
    assert(get("/health").body() == """{"status":"ok"}""")
  }

  test("test_root_or_app_specific_smoke") {
    val r = get("/")
    assert(r.statusCode() == 200)
    assert(r.body() == "scala2-bazel")
  }
}
