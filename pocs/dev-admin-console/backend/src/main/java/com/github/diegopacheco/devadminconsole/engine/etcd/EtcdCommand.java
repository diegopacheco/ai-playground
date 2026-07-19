package com.github.diegopacheco.devadminconsole.engine.etcd;

public record EtcdCommand(String operation, String key, String rangeEnd, boolean prefix, int limit) {}
