package com.diegopacheco.temporalpoc.config;

import com.diegopacheco.temporalpoc.workflow.CompanyResearchWorkflowImpl;
import io.temporal.client.WorkflowClient;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class TemporalConfig {
    private static final Logger log = LoggerFactory.getLogger(TemporalConfig.class);

    @Bean
    WorkflowServiceStubs workflowServiceStubs(@Value("${temporal.address}") String address) {
        log.info("creating temporal service stubs address={}", address);
        return WorkflowServiceStubs.newServiceStubs(
                io.temporal.serviceclient.WorkflowServiceStubsOptions.newBuilder()
                        .setTarget(address)
                        .build()
        );
    }

    @Bean
    WorkflowClient workflowClient(WorkflowServiceStubs stubs) {
        log.info("creating temporal workflow client");
        return WorkflowClient.newInstance(stubs);
    }

    @Bean
    WorkerFactory workerFactory(WorkflowClient client, WorkerActivities activities, @Value("${temporal.task-queue}") String taskQueue) {
        log.info("creating temporal worker factory taskQueue={}", taskQueue);
        WorkerFactory factory = WorkerFactory.newInstance(client);
        Worker worker = factory.newWorker(taskQueue);
        log.info("registering workflow implementation type={}", CompanyResearchWorkflowImpl.class.getName());
        worker.registerWorkflowImplementationTypes(CompanyResearchWorkflowImpl.class);
        log.info("registering activity implementations stock={} news={} decision={}", activities.stock().getClass().getName(), activities.news().getClass().getName(), activities.decision().getClass().getName());
        worker.registerActivitiesImplementations(activities.stock(), activities.news(), activities.decision());
        factory.start();
        log.info("temporal worker factory started taskQueue={}", taskQueue);
        return factory;
    }
}
