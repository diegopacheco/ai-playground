package com.github.diegopacheco.decisions.logging;

import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

public final class AppLogger {

    private static final String SERVICE = "decisions-service";

    private final Logger log;

    private AppLogger(Class<?> owner) {
        this.log = LoggerFactory.getLogger(owner);
    }

    public static AppLogger of(Class<?> owner) {
        return new AppLogger(owner);
    }

    public void info(String event, Map<String, Object> fields) {
        log.atInfo().setMessage(line(event, fields)).log();
    }

    public void error(String event, Throwable cause, Map<String, Object> fields) {
        log.atError().setMessage(line(event, fields)).setCause(cause).log();
    }

    private static String line(String event, Map<String, Object> fields) {
        return "%s service=%s correlationId=%s path=%s userId=%s fields=%s".formatted(event, SERVICE,
                MDC.get(CorrelationId.KEY), MDC.get("path"), "anonymous", fields);
    }
}
