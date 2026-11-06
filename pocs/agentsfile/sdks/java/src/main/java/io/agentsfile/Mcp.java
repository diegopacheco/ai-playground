package io.agentsfile;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public record Mcp(
        String name,
        Transport transport,
        Optional<String> description,
        Optional<String> command,
        List<String> args,
        Map<String, String> env,
        Optional<String> url,
        Map<String, String> headers) {
    public Mcp {
        args = List.copyOf(args);
        env = Collections.unmodifiableMap(new LinkedHashMap<>(env));
        headers = Collections.unmodifiableMap(new LinkedHashMap<>(headers));
    }
}
