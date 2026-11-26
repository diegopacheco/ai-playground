package com.github.diegopacheco.decisions.logging;

import java.io.IOException;
import java.util.Optional;
import java.util.UUID;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.slf4j.MDC;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
public class CorrelationIdFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(HttpServletRequest req, HttpServletResponse res, FilterChain chain)
            throws ServletException, IOException {
        String id = Optional.ofNullable(req.getHeader(CorrelationId.HEADER)).orElseGet(() -> UUID.randomUUID().toString());
        MDC.put(CorrelationId.KEY, id);
        MDC.put("path", req.getRequestURI());
        res.setHeader(CorrelationId.HEADER, id);
        try {
            chain.doFilter(req, res);
        } finally {
            MDC.remove(CorrelationId.KEY);
            MDC.remove("path");
        }
    }
}
