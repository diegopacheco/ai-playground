package com.diegopacheco.temporalpoc.service;

public class CodexCliException extends RuntimeException {
    public CodexCliException(String message) {
        super(message);
    }

    public CodexCliException(String message, Throwable cause) {
        super(message, cause);
    }
}
