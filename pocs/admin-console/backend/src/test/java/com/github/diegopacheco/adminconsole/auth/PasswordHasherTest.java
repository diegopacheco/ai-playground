package com.github.diegopacheco.adminconsole.auth;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class PasswordHasherTest {
    private final PasswordHasher hasher = new PasswordHasher();

    @Test
    void acceptsTheCorrectPassword() {
        byte[] salt = hasher.salt();
        assertThat(hasher.matches("correct-horse", salt, hasher.hash("correct-horse", salt))).isTrue();
    }

    @Test
    void rejectsTheWrongPassword() {
        byte[] salt = hasher.salt();
        assertThat(hasher.matches("wrong", salt, hasher.hash("correct-horse", salt))).isFalse();
    }

    @Test
    void saltsEachPasswordSoIdenticalPasswordsDoNotShareAHashAcrossUsers() {
        byte[] first = hasher.salt();
        byte[] second = hasher.salt();
        assertThat(first).isNotEqualTo(second);
        assertThat(hasher.hash("same", first)).isNotEqualTo(hasher.hash("same", second));
    }

    @Test
    void rejectsMissingCredentialsRatherThanThrowingSoALoginProbeCannotCrashTheFilter() {
        byte[] salt = hasher.salt();
        byte[] hash = hasher.hash("password", salt);
        assertThat(hasher.matches(null, salt, hash)).isFalse();
        assertThat(hasher.matches("password", null, hash)).isFalse();
        assertThat(hasher.matches("password", salt, null)).isFalse();
    }
}
