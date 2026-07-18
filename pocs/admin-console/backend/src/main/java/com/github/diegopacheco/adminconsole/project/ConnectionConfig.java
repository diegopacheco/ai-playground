package com.github.diegopacheco.adminconsole.project;

public record ConnectionConfig(Long id, Long projectId, String name, ConnectionKind kind, String host, int port,
                               String database, String keyspace, String datacenter, String username, String password,
                               String options, String createdBy) {
    public ConnectionConfig withPassword(String value) {
        return new ConnectionConfig(id, projectId, name, kind, host, port, database, keyspace, datacenter, username,
                value, options, createdBy);
    }
}
