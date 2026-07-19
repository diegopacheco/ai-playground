package com.github.diegopacheco.devadminconsole.user;

import java.time.Instant;

public record User(Long id, String username, byte[] passwordHash, byte[] passwordSalt, String role, boolean enabled,
                   Instant createdAt, Instant lastLoginAt) {
    public static final String ADMIN = "admin";
    public static final String USER = "user";

    public boolean isAdmin() {
        return ADMIN.equals(role);
    }
}
