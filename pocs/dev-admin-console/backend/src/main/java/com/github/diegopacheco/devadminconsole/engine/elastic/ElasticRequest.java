package com.github.diegopacheco.devadminconsole.engine.elastic;

public record ElasticRequest(String method, String path, String body) {}
