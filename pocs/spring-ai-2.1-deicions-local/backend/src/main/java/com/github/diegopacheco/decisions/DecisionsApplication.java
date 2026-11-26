package com.github.diegopacheco.decisions;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class DecisionsApplication {

    public static void main(String[] args) {
        SpringApplication.run(DecisionsApplication.class, args);
    }
}
