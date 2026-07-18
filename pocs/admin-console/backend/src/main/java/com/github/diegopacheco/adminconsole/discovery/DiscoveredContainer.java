package com.github.diegopacheco.adminconsole.discovery;

import com.github.diegopacheco.adminconsole.project.ConnectionKind;

public record DiscoveredContainer(String id, String name, String image, ConnectionKind kind, int hostPort,
                                  int containerPort, String database, String keyspace, String username,
                                  String password, boolean importable, String reason) {

    public static DiscoveredContainer unreachable(String id, String name, String image, ConnectionKind kind,
                                                  int containerPort, String reason) {
        return new DiscoveredContainer(id, name, image, kind, 0, containerPort, null, null, null, null, false, reason);
    }
}
