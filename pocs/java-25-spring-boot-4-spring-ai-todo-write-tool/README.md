# Spring AI - TODO Write Tool

https://spring.io/blog/2026/01/20/spring-ai-agentic-patterns-3-todowrite

<img src="todo-write-flow.png" width="600" />

## Result

Server:
```
[INFO] --- spring-boot:4.0.2:run (default-cli) @ java-25-spring-boot-4-spring-ai-todo-write-tool ---
[INFO] Attaching agents: []

  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/

 :: Spring Boot ::                (v4.0.2)

2026-02-19T22:48:21.983-08:00  INFO 27155 --- [           main] c.g.d.sandboxspring.Application          : Starting Application using Java 25 with PID 27155 (/Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-todo-write-tool/target/classes started by diegopacheco in /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-todo-write-tool)
2026-02-19T22:48:21.985-08:00  INFO 27155 --- [           main] c.g.d.sandboxspring.Application          : No active profile set, falling back to 1 default profile: "default"
2026-02-19T22:48:22.513-08:00  INFO 27155 --- [           main] o.s.boot.tomcat.TomcatWebServer          : Tomcat initialized with port 8081 (http)
2026-02-19T22:48:22.519-08:00  INFO 27155 --- [           main] o.apache.catalina.core.StandardService   : Starting service [Tomcat]
2026-02-19T22:48:22.519-08:00  INFO 27155 --- [           main] o.apache.catalina.core.StandardEngine    : Starting Servlet engine: [Apache Tomcat/11.0.15]
2026-02-19T22:48:22.544-08:00  INFO 27155 --- [           main] b.w.c.s.WebApplicationContextInitializer : Root WebApplicationContext: initialization completed in 534 ms
2026-02-19T22:48:23.041-08:00  INFO 27155 --- [           main] o.s.b.a.e.web.EndpointLinksResolver      : Exposing 1 endpoint beneath base path '/actuator'
2026-02-19T22:48:23.070-08:00  INFO 27155 --- [           main] o.s.boot.tomcat.TomcatWebServer          : Tomcat started on port 8081 (http) with context path '/'
2026-02-19T22:48:23.075-08:00  INFO 27155 --- [           main] c.g.d.sandboxspring.Application          : Started Application in 1.264 seconds (process running for 1.435)
2026-02-19T22:48:27.844-08:00  INFO 27155 --- [nio-8081-exec-1] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring DispatcherServlet 'dispatcherServlet'
2026-02-19T22:48:27.844-08:00  INFO 27155 --- [nio-8081-exec-1] o.s.web.servlet.DispatcherServlet        : Initializing Servlet 'dispatcherServlet'
2026-02-19T22:48:27.845-08:00  INFO 27155 --- [nio-8081-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 1 ms
WARNING: A restricted method in java.lang.System has been called
WARNING: java.lang.System::loadLibrary has been called by io.netty.util.internal.NativeLibraryUtil in an unnamed module (file:/Users/diegopacheco/.m2/repository/io/netty/netty-common/4.2.9.Final/netty-common-4.2.9.Final.jar)
WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
WARNING: Restricted methods will be blocked in a future release unless native access is enabled

2026-02-19T22:48:28.460-08:00 ERROR 27155 --- [nio-8081-exec-2] i.n.r.d.DnsServerAddressStreamProviders  : Unable to load io.netty.resolver.dns.macos.MacOSDnsServerAddressStreamProvider, fallback to system defaults. This may result in incorrect DNS resolutions on MacOS. Check whether you have a dependency on 'io.netty:netty-resolver-dns-native-macos'. Use DEBUG level to see the full stack: java.lang.UnsatisfiedLinkError: failed to load the required native library

Progress: 0/4 tasks completed (0%)
  [ ] List the top 3 Spring Boot features
  [ ] Explain each of the top 3 Spring Boot features
  [ ] Provide a code snippet for each feature
  [ ] Write a summary of the top 3 Spring Boot features

Progress: 0/4 tasks completed (0%)
  [→] List the top 3 Spring Boot features
  [ ] Explain each of the top 3 Spring Boot features
  [ ] Provide a code snippet for each feature
  [ ] Write a summary of the top 3 Spring Boot features

Progress: 1/4 tasks completed (25%)
  [✓] List the top 3 Spring Boot features
  [→] Explain each of the top 3 Spring Boot features
  [ ] Provide a code snippet for each feature
  [ ] Write a summary of the top 3 Spring Boot features

Progress: 2/4 tasks completed (50%)
  [✓] List the top 3 Spring Boot features
  [✓] Explain each of the top 3 Spring Boot features
  [→] Provide a code snippet for each feature
  [ ] Write a summary of the top 3 Spring Boot features

Progress: 3/4 tasks completed (75%)
  [✓] List the top 3 Spring Boot features
  [✓] Explain each of the top 3 Spring Boot features
  [✓] Provide a code snippet for each feature
  [→] Write a summary of the top 3 Spring Boot features

Progress: 0/4 tasks completed (0%)
  [ ] Identify the top 5 Java design patterns
  [ ] Describe each design pattern briefly
  [ ] Provide a real-world use case for each design pattern
  [ ] Write a summary of the top 5 Java design patterns

Progress: 0/4 tasks completed (0%)
  [→] Identify the top 5 Java design patterns
  [ ] Describe each design pattern briefly
  [ ] Provide a real-world use case for each design pattern
  [ ] Write a summary of the top 5 Java design patterns

Progress: 1/4 tasks completed (25%)
  [✓] Identify the top 5 Java design patterns
  [→] Describe each design pattern briefly
  [ ] Provide a real-world use case for each design pattern
  [ ] Write a summary of the top 5 Java design patterns

Progress: 2/4 tasks completed (50%)
  [✓] Identify the top 5 Java design patterns
  [✓] Describe each design pattern briefly
  [→] Provide a real-world use case for each design pattern
  [ ] Write a summary of the top 5 Java design patterns

Progress: 3/4 tasks completed (75%)
  [✓] Identify the top 5 Java design patterns
  [✓] Describe each design pattern briefly
  [✓] Provide a real-world use case for each design pattern
  [→] Write a summary of the top 5 Java design patterns
```

Test client
```
❯ ./test.sh
=== Testing root endpoint ===
Greetings from Spring Boot with TodoWrite Tool!

=== Testing TodoWrite with complex multi-step task ===
### Summary

Spring Boot is a powerful framework that simplifies the development of Java applications by providing a suite of features to enhance productivity. The **Auto-Configuration** feature reduces the need for manual setup by automatically configuring Spring applications based on the dependencies present in the project, allowing developers to focus on writing application logic rather than boilerplate code. **Spring Boot Starter Projects** streamline the process of dependency management by offering pre-packaged templates for various functionalities, eliminating the hassle of configuring each dependency manually. Finally, **Spring Boot Actuator** equips developers with tools to monitor and manage their applications in production environments, providing essential health checks and metrics to ensure applications run smoothly. Together, these features make Spring Boot an essential tool for modern Java development, offering simplicity, ease of use, and robust support for building enterprise-grade applications.
```