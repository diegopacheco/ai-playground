package bp

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.web.bind.annotation.{GetMapping, RestController}

@SpringBootApplication
@RestController
class App {
  @GetMapping(path = Array("/"))
  def root(): String = "java25-scala3-sbt-sb4"
}

object App {
  def main(args: Array[String]): Unit = {
    SpringApplication.run(classOf[App], args*)
  }
}
