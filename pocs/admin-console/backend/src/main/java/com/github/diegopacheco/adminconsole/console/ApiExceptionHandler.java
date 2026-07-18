package com.github.diegopacheco.adminconsole.console;

import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class ApiExceptionHandler {
    @ExceptionHandler(com.github.diegopacheco.adminconsole.engine.ReadOnlyViolation.class)
    public ResponseEntity<Map<String, Object>> readOnly(
            com.github.diegopacheco.adminconsole.engine.ReadOnlyViolation error) {
        return ResponseEntity.badRequest().body(Map.of("error", message(error), "readOnlyViolation", true));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> badRequest(IllegalArgumentException error) {
        return ResponseEntity.badRequest().body(Map.of("error", message(error)));
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<Map<String, Object>> conflict(IllegalStateException error) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of("error", message(error)));
    }

    @ExceptionHandler(RuntimeException.class)
    public ResponseEntity<Map<String, Object>> unexpected(RuntimeException error) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of("error", rootCause(error)));
    }

    private String rootCause(Throwable error) {
        Throwable cause = error;
        while (cause.getCause() != null && cause.getCause() != cause) {
            cause = cause.getCause();
        }
        String text = cause.getMessage() == null ? cause.getClass().getSimpleName() : cause.getMessage();
        return text.length() > 400 ? text.substring(0, 400) + "…" : text;
    }

    private String message(Exception error) {
        return error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
    }
}
