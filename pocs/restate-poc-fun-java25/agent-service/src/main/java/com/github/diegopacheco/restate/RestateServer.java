package com.github.diegopacheco.restate;

import com.github.diegopacheco.restate.config.AgentConfig;
import com.github.diegopacheco.restate.config.ServicePort;
import com.github.diegopacheco.restate.workflow.AgentWorkflow;
import dev.restate.sdk.endpoint.Endpoint;
import dev.restate.sdk.http.vertx.RestateHttpServer;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

@Component
public class RestateServer implements ApplicationRunner {

    private final AgentWorkflow workflow;

    public RestateServer(AgentWorkflow workflow) {
        this.workflow = workflow;
    }

    @Override
    public void run(ApplicationArguments args) {
        AgentConfig config = AgentConfig.from(System::getenv);
        int port = ServicePort.from(System::getenv);
        RestateHttpServer.listen(Endpoint.bind(workflow).build(), port);
        System.out.printf("agent-service on port %d using %s %s%n", port, config.provider().cli(), config.model());
    }
}
