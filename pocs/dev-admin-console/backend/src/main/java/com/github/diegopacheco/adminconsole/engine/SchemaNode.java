package com.github.diegopacheco.adminconsole.engine;

import java.util.List;

public record SchemaNode(String name, String kind, String detail, List<SchemaNode> children) {
    public static SchemaNode leaf(String name, String kind, String detail) {
        return new SchemaNode(name, kind, detail, List.of());
    }
}
