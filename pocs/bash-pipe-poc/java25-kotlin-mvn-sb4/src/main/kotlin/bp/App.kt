package bp

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@SpringBootApplication
@RestController
class App {
    @GetMapping("/")
    fun root(): String = "java25-kotlin-mvn-sb4"
}

fun main(args: Array<String>) {
    SpringApplication.run(App::class.java, *args)
}
