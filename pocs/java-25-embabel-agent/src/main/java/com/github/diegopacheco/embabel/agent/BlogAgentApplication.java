package com.github.diegopacheco.embabel.agent;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import com.embabel.agent.config.annotation.EnableAgents;

@SpringBootApplication
@EnableAgents
public class BlogAgentApplication {
    public static void main(String[] args) {
        SpringApplication.run(BlogAgentApplication.class, args);
    }
}
