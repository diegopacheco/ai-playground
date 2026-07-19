package com.github.diegopacheco.devadminconsole.engine;

public class ReadOnlyViolation extends RuntimeException {
    public ReadOnlyViolation(String message) {
        super(message);
    }
}
