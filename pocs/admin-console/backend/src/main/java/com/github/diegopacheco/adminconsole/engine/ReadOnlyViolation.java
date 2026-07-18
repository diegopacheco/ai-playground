package com.github.diegopacheco.adminconsole.engine;

public class ReadOnlyViolation extends RuntimeException {
    public ReadOnlyViolation(String message) {
        super(message);
    }
}
