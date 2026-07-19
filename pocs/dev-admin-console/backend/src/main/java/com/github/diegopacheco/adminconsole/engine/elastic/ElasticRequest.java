package com.github.diegopacheco.adminconsole.engine.elastic;

public record ElasticRequest(String method, String path, String body) {}
