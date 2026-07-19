package com.github.diegopacheco.devadminconsole.engine;

import com.github.diegopacheco.devadminconsole.project.ConnectionKind;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.springframework.stereotype.Component;

@Component
public class EngineRegistry {
    private final Map<ConnectionKind, Engine> engines;

    public EngineRegistry(List<Engine> engines) {
        this.engines = engines.stream().collect(Collectors.toMap(Engine::kind, Function.identity()));
    }

    public Engine of(ConnectionKind kind) {
        Engine engine = engines.get(kind);
        if (engine == null) {
            throw new IllegalArgumentException("no engine for kind: " + kind.wireName());
        }
        return engine;
    }
}
