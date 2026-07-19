package com.github.diegopacheco.devadminconsole.engine.kafka;

public record KafkaCommand(String operation, String target, Integer partition, String from, Long offset, int limit) {}
