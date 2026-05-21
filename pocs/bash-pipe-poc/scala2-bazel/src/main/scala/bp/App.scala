package bp

import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}

import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets

object App {

  def main(args: Array[String]): Unit = {
    val port = sys.env.get("PORT").map(_.toInt).getOrElse(8086)
    runOn(port)
    Thread.currentThread().join()
  }

  def runOn(port: Int): HttpServer = {
    val server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0)
    server.createContext("/health", new Handler(200, """{"status":"ok"}""", "application/json"))
    server.createContext("/", new Handler(200, "scala2-bazel", "text/plain"))
    server.setExecutor(null)
    server.start()
    server
  }

  class Handler(status: Int, body: String, contentType: String) extends HttpHandler {
    override def handle(ex: HttpExchange): Unit = {
      val b = body.getBytes(StandardCharsets.UTF_8)
      ex.getResponseHeaders.add("Content-Type", contentType)
      ex.sendResponseHeaders(status, b.length.toLong)
      val os = ex.getResponseBody
      os.write(b)
      os.close()
    }
  }
}
