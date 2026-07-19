package com.github.diegopacheco.adminconsole.engine;

import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.List;

public interface Engine {
    ConnectionKind kind();

    List<SchemaNode> schema(ConnectionConfig config);

    QueryResult query(ConnectionConfig config, String statement, PageRequest page);

    void assertReadOnly(String statement);

    default boolean ping(ConnectionConfig config) {
        schema(config);
        return true;
    }
}
