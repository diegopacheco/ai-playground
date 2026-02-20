# Spring AI - AskUserQuestion Tool

https://spring.io/blog/2026/01/16/spring-ai-ask-user-question-tool

<img src="ask-user-question-tool-flow.png" width="600" alt="AskUserQuestion Tool Flow">

## Result

```
export OPENAI_API_KEY=your_api_key_here
./run.sh
```

```
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  1.260 s
[INFO] Finished at: 2026-02-19T22:09:55-08:00
[INFO] ------------------------------------------------------------------------

  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/

 :: Spring Boot ::                (v4.0.2)

2026-02-19T22:09:56.345-08:00  INFO 80665 --- [ask-user-tool] [           main] c.g.d.s.AskUserToolApplication           : Starting AskUserToolApplication v1.0-SNAPSHOT using Java 25 with PID 80665 (/Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-ask-user-tool/target/java-25-spring-boot-4-spring-ai-ask-user-tool-1.0-SNAPSHOT.jar started by diegopacheco in /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-ask-user-tool)
2026-02-19T22:09:56.347-08:00  INFO 80665 --- [ask-user-tool] [           main] c.g.d.s.AskUserToolApplication           : No active profile set, falling back to 1 default profile: "default"
2026-02-19T22:09:57.090-08:00  INFO 80665 --- [ask-user-tool] [           main] c.g.d.s.AskUserToolApplication           : Started AskUserToolApplication in 0.994 seconds (process running for 1.279)

=== Spring AI Ask User Question Tool ===

WARNING: A restricted method in java.lang.System has been called
WARNING: java.lang.System::loadLibrary has been called by io.netty.util.internal.NativeLibraryUtil in an unnamed module (jar:nested:/Users/diegopacheco/git/diegopacheco/ai-playground/pocs/java-25-spring-boot-4-spring-ai-ask-user-tool/target/java-25-spring-boot-4-spring-ai-ask-user-tool-1.0-SNAPSHOT.jar/!BOOT-INF/lib/netty-common-4.2.9.Final.jar!/)
WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
WARNING: Restricted methods will be blocked in a future release unless native access is enabled
```

Call the Agent:

```
./test.sh
```

Result:
```

```