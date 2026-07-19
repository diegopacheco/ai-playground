package com.github.diegopacheco.adminconsole.user;

import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.constraints.NotBlank;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/users")
public class UserController {
    public record CreateRequest(@NotBlank String username, @NotBlank String password, @NotBlank String role) {}
    public record UpdateRequest(@NotBlank String role, boolean enabled) {}
    public record PasswordRequest(@NotBlank String password) {}

    private final UserService users;

    public UserController(UserService users) {
        this.users = users;
    }

    @GetMapping
    @Operation(summary = "List all users")
    public List<Map<String, Object>> list() {
        return users.findAll().stream().map(UserController::view).toList();
    }

    @PostMapping
    @Operation(summary = "Create a user")
    public Map<String, Object> create(@RequestBody CreateRequest request) {
        return view(users.create(request.username(), request.password(), request.role()));
    }

    @PutMapping("/{id}")
    @Operation(summary = "Change a user role or enabled state")
    public Map<String, Object> update(@PathVariable long id, @RequestBody UpdateRequest request) {
        users.updateRoleAndEnabled(id, request.role(), request.enabled());
        return Map.of("updated", true);
    }

    @PostMapping("/{id}/password")
    @Operation(summary = "Reset the password of a user")
    public Map<String, Object> password(@PathVariable long id, @RequestBody PasswordRequest request) {
        users.changePassword(id, request.password());
        return Map.of("changed", true);
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a user")
    public Map<String, Object> delete(@PathVariable long id) {
        users.delete(id);
        return Map.of("deleted", true);
    }

    private static Map<String, Object> view(User user) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", user.id());
        body.put("username", user.username());
        body.put("role", user.role());
        body.put("enabled", user.enabled());
        body.put("createdAt", user.createdAt());
        body.put("lastLoginAt", user.lastLoginAt());
        return body;
    }
}
