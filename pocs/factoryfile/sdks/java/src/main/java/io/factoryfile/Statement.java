package io.factoryfile;

import java.util.List;

public sealed interface Statement {
    String name();

    int line();

    int col();

    record Attribute(String name, Value value, int line, int col) implements Statement {
    }

    record Block(String name, String label, List<Statement> body, int line, int col) implements Statement {
        public Block {
            body = List.copyOf(body);
        }
    }
}
