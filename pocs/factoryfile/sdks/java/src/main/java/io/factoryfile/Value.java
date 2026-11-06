package io.factoryfile;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public sealed interface Value {
    int line();

    int col();

    record StringValue(String value, int line, int col) implements Value {
    }

    record ListValue(List<StringValue> items, int line, int col) implements Value {
        public ListValue {
            items = List.copyOf(items);
        }
    }

    record MapValue(Map<String, StringValue> entries, int line, int col) implements Value {
        public MapValue {
            entries = Collections.unmodifiableMap(new LinkedHashMap<>(entries));
        }
    }
}
