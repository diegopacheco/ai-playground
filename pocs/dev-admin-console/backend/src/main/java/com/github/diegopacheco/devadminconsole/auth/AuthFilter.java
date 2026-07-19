package com.github.diegopacheco.devadminconsole.auth;

import com.github.diegopacheco.devadminconsole.user.User;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 10)
public class AuthFilter extends OncePerRequestFilter {
    public static final String COOKIE = "dev_admin_console_token";

    private final Jwt jwt;

    public AuthFilter(Jwt jwt) {
        this.jwt = jwt;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        String path = request.getRequestURI();
        if ("OPTIONS".equals(request.getMethod()) || !protectedPath(path)) {
            chain.doFilter(request, response);
            return;
        }
        Jwt.Principal principal = jwt.verify(token(request));
        if (principal == null) {
            deny(response, HttpStatus.UNAUTHORIZED, "login required");
            return;
        }
        if (requiresAdmin(path, request.getMethod()) && !User.ADMIN.equals(principal.role())) {
            deny(response, HttpStatus.FORBIDDEN, "admin role required");
            return;
        }
        request.setAttribute(CurrentUser.ATTRIBUTE, principal);
        chain.doFilter(request, response);
    }

    boolean protectedPath(String path) {
        if (path.equals("/api/auth/login") || path.startsWith("/actuator/health")) {
            return false;
        }
        return path.startsWith("/api/")
                || path.equals("/swagger")
                || path.startsWith("/swagger-ui")
                || path.startsWith("/v3/api-docs");
    }

    boolean requiresAdmin(String path, String method) {
        if (path.startsWith("/api/users") || path.startsWith("/api/audit")) {
            return true;
        }
        if (path.startsWith("/api/projects") && !"GET".equals(method)) {
            return !path.contains("/saved");
        }
        return false;
    }

    private String token(HttpServletRequest request) {
        String authorization = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (authorization != null && authorization.startsWith("Bearer ")) {
            return authorization.substring(7);
        }
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if (COOKIE.equals(cookie.getName())) {
                    return cookie.getValue();
                }
            }
        }
        return "";
    }

    private void deny(HttpServletResponse response, HttpStatus status, String message) throws IOException {
        response.setStatus(status.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.getOutputStream().write(("{\"error\":\"" + message + "\"}").getBytes(StandardCharsets.UTF_8));
    }
}
