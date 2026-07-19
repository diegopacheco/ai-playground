package com.github.diegopacheco.adminconsole.engine.kafka;

public record KafkaCommand(String operation, String target, Integer partition, String from, Long offset, int limit) {}
