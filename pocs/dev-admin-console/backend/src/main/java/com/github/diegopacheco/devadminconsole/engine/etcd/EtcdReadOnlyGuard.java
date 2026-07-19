package com.github.diegopacheco.devadminconsole.engine.etcd;

import com.github.diegopacheco.devadminconsole.engine.ReadOnlyViolation;
import java.util.Set;
import org.springframework.stereotype.Component;

@Component
public class EtcdReadOnlyGuard {
    private static final Set<String> ALLOWED = Set.of("get", "range");

    public void assertReadOnly(EtcdCommand command) {
        if (!ALLOWED.contains(command.operation())) {
            throw new ReadOnlyViolation(command.operation() + " is not a read operation, allowed: get, range");
        }
        if (command.key() == null) {
            throw new ReadOnlyViolation("a key is required");
        }
    }
}
