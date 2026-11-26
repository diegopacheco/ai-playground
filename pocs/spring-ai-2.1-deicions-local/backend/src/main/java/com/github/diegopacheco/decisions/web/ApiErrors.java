package com.github.diegopacheco.decisions.web;

import java.util.Map;

import com.github.diegopacheco.decisions.logging.AppLogger;

import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class ApiErrors {

    private static final AppLogger log = AppLogger.of(ApiErrors.class);

    @ExceptionHandler(MethodArgumentNotValidException.class)
    ProblemDetail invalid(MethodArgumentNotValidException e) {
        String detail = e.getBindingResult().getFieldErrors().stream()
                .map(f -> f.getField() + " " + f.getDefaultMessage()).sorted().reduce((a, b) -> a + ", " + b)
                .orElse("invalid request");
        return ProblemDetail.forStatusAndDetail(HttpStatus.BAD_REQUEST, detail);
    }

    @ExceptionHandler(RuntimeException.class)
    ProblemDetail decisionFailed(RuntimeException e) {
        log.error("decision call failed", e, Map.of("type", e.getClass().getSimpleName()));
        return ProblemDetail.forStatusAndDetail(HttpStatus.BAD_GATEWAY,
                "The decision model could not answer: " + e.getMessage());
    }
}
