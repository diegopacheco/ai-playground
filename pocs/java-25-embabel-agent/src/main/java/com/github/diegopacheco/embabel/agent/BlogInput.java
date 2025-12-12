package com.github.diegopacheco.embabel.agent;

public record BlogInput(String topic, String keywords, String tone) {
    public BlogInput(String topic) {
        this(topic, "", "professional");
    }
}
