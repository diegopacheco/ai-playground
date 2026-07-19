package com.github.diegopacheco.devadminconsole.engine.kafka;

import com.github.diegopacheco.devadminconsole.engine.ReadOnlyViolation;
import java.util.Set;
import org.springframework.stereotype.Component;

@Component
public class KafkaReadOnlyGuard {
    private static final Set<String> ALLOWED = Set.of("list", "describe", "offsets", "consume");

    public void assertOperationAllowed(String operation) {
        if (!ALLOWED.contains(operation.toLowerCase())) {
            throw new ReadOnlyViolation(operation
                    + " is not a read operation, allowed: list, describe, offsets, consume");
        }
    }

    public void assertReadOnly(KafkaCommand command) {
        assertOperationAllowed(command.operation());
    }
}
