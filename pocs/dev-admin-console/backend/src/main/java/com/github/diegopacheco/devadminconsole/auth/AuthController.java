package com.github.diegopacheco.devadminconsole.auth;

import com.github.diegopacheco.devadminconsole.user.User;
import com.github.diegopacheco.devadminconsole.user.UserService;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.constraints.NotBlank;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    public record LoginRequest(@NotBlank String username, @NotBlank String password) {}
    public record PasswordRequest(@NotBlank String password) {}

    private final UserService users;
    private final Jwt jwt;
    private final CurrentUser current;
    private final String bootstrapUsername;
    private final String bootstrapPassword;

    public AuthController(UserService users, Jwt jwt, CurrentUser current,
                          @Value("${app.admin.bootstrap-username}") String bootstrapUsername,
                          @Value("${app.admin.bootstrap-password}") String bootstrapPassword) {
        this.users = users;
        this.jwt = jwt;
        this.current = current;
        this.bootstrapUsername = bootstrapUsername;
        this.bootstrapPassword = bootstrapPassword;
    }

    @PostMapping("/login")
    @Operation(summary = "Exchange username and password for a session cookie")
    public ResponseEntity<Map<String, Object>> login(@RequestBody LoginRequest request) {
        Optional<User> user = users.authenticate(request.username(), request.password());
        if (user.isEmpty()) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("error", "invalid credentials"));
        }
        String token = jwt.issue(user.get().username(), user.get().role());
        ResponseCookie cookie = ResponseCookie.from(AuthFilter.COOKIE, token)
                .httpOnly(true).sameSite("Lax").path("/").maxAge(jwt.lifetimeSeconds()).build();
        return ResponseEntity.ok().header(HttpHeaders.SET_COOKIE, cookie.toString()).body(session(user.get()));
    }

    @PostMapping("/logout")
    @Operation(summary = "Clear the session cookie")
    public ResponseEntity<Map<String, Object>> logout() {
        ResponseCookie cookie = ResponseCookie.from(AuthFilter.COOKIE, "")
                .httpOnly(true).sameSite("Lax").path("/").maxAge(0).build();
        return ResponseEntity.ok().header(HttpHeaders.SET_COOKIE, cookie.toString()).body(Map.of("loggedOut", true));
    }

    @GetMapping("/session")
    @Operation(summary = "Describe the current session")
    public ResponseEntity<Map<String, Object>> session() {
        return users.findByUsername(current.username())
                .map(user -> ResponseEntity.ok(session(user)))
                .orElseGet(() -> ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("error", "login required")));
    }

    @PostMapping("/password")
    @Operation(summary = "Change the password of the current user")
    public ResponseEntity<Map<String, Object>> password(@RequestBody PasswordRequest request) {
        User user = users.findByUsername(current.username()).orElseThrow();
        users.changePassword(user.id(), request.password());
        return ResponseEntity.ok(Map.of("changed", true));
    }

    private Map<String, Object> session(User user) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("username", user.username());
        body.put("role", user.role());
        if (user.isAdmin()) {
            body.put("usingBootstrapPassword", users.usingBootstrapPassword(bootstrapUsername, bootstrapPassword));
        }
        return body;
    }
}
