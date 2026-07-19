package com.github.diegopacheco.adminconsole.engine.etcd;

public record EtcdCommand(String operation, String key, String rangeEnd, boolean prefix, int limit) {}
