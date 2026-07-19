package com.github.diegopacheco.adminconsole.engine.redis;

import com.github.diegopacheco.adminconsole.engine.ReadOnlyViolation;
import java.util.Set;
import org.springframework.stereotype.Component;

@Component
public class RedisReadOnlyGuard {
    private static final Set<String> ALWAYS_DENIED = Set.of(
            "EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO", "FUNCTION", "SCRIPT",
            "SUBSCRIBE", "PSUBSCRIBE", "SSUBSCRIBE", "MONITOR", "SHUTDOWN", "DEBUG", "RESET", "SWAPDB",
            "MULTI", "EXEC", "DISCARD", "WATCH");

    public void assertReadOnly(String command, Set<String> flags) {
        String operation = command.toUpperCase();
        if (ALWAYS_DENIED.contains(operation)) {
            throw new ReadOnlyViolation(operation + " can execute or subscribe and is not allowed");
        }
        if (flags == null || flags.isEmpty()) {
            throw new ReadOnlyViolation("unknown command: " + operation);
        }
        if (flags.contains("write")) {
            throw new ReadOnlyViolation(operation + " is a write command and is not allowed");
        }
        if (flags.contains("admin")) {
            throw new ReadOnlyViolation(operation + " is an admin command and is not allowed");
        }
        if (!flags.contains("readonly")) {
            throw new ReadOnlyViolation(operation + " is not a read-only command");
        }
    }
}
