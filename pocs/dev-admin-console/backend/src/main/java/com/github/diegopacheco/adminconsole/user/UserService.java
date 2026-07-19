package com.github.diegopacheco.adminconsole.user;

import com.github.diegopacheco.adminconsole.auth.PasswordHasher;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    private final UserRepository users;
    private final PasswordHasher hasher;

    public UserService(UserRepository users, PasswordHasher hasher) {
        this.users = users;
        this.hasher = hasher;
    }

    public Optional<User> authenticate(String username, String password) {
        Optional<User> found = users.findByUsername(username);
        if (found.isEmpty() || !found.get().enabled()) {
            return Optional.empty();
        }
        User user = found.get();
        if (!hasher.matches(password, user.passwordSalt(), user.passwordHash())) {
            return Optional.empty();
        }
        users.touchLogin(user.id());
        return found;
    }

    public User create(String username, String password, String role) {
        if (username == null || username.isBlank()) {
            throw new IllegalArgumentException("username is required");
        }
        if (password == null || password.length() < 4) {
            throw new IllegalArgumentException("password must be at least 4 characters");
        }
        if (!User.ADMIN.equals(role) && !User.USER.equals(role)) {
            throw new IllegalArgumentException("role must be admin or user");
        }
        if (users.findByUsername(username).isPresent()) {
            throw new IllegalArgumentException("username already exists");
        }
        byte[] salt = hasher.salt();
        return users.create(username, hasher.hash(password, salt), salt, role);
    }

    public void changePassword(long id, String password) {
        if (password == null || password.length() < 4) {
            throw new IllegalArgumentException("password must be at least 4 characters");
        }
        byte[] salt = hasher.salt();
        users.updatePassword(id, hasher.hash(password, salt), salt);
    }

    public void updateRoleAndEnabled(long id, String role, boolean enabled) {
        if (!User.ADMIN.equals(role) && !User.USER.equals(role)) {
            throw new IllegalArgumentException("role must be admin or user");
        }
        User user = users.findById(id).orElseThrow(() -> new IllegalArgumentException("user not found"));
        if (user.isAdmin() && (!User.ADMIN.equals(role) || !enabled) && remainingAdmins(id) == 0) {
            throw new IllegalArgumentException("the last admin cannot be demoted or disabled");
        }
        users.updateRoleAndEnabled(id, role, enabled);
    }

    public void delete(long id) {
        User user = users.findById(id).orElseThrow(() -> new IllegalArgumentException("user not found"));
        if (user.isAdmin() && remainingAdmins(id) == 0) {
            throw new IllegalArgumentException("the last admin cannot be deleted");
        }
        users.delete(id);
    }

    public List<User> findAll() {
        return users.findAll();
    }

    public Optional<User> findByUsername(String username) {
        return users.findByUsername(username);
    }

    public boolean usingBootstrapPassword(String bootstrapUsername, String bootstrapPassword) {
        return users.findByUsername(bootstrapUsername)
                .map(user -> hasher.matches(bootstrapPassword, user.passwordSalt(), user.passwordHash()))
                .orElse(false);
    }

    private long remainingAdmins(long excludedId) {
        return users.findAll().stream()
                .filter(User::isAdmin)
                .filter(User::enabled)
                .filter(user -> user.id() != excludedId)
                .count();
    }
}
