package com.github.diegopacheco.devadminconsole.auth;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import java.util.Base64;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class JwtTest {
    private Jwt jwt;

    @BeforeEach
    void setUp() {
        byte[] key = new byte[32];
        Arrays.fill(key, (byte) 3);
        jwt = new Jwt(purpose -> key);
    }

    @Test
    void carriesUsernameAndRoleSoAuthorizationDoesNotNeedADatabaseReadPerRequest() {
        Jwt.Principal principal = jwt.verify(jwt.issue("diego", "admin"));
        assertThat(principal).isNotNull();
        assertThat(principal.username()).isEqualTo("diego");
        assertThat(principal.role()).isEqualTo("admin");
    }

    @Test
    void rejectsATokenWhoseRoleClaimWasEditedSoAUserCannotPromoteThemselves() {
        String token = jwt.issue("diego", "user");
        String[] parts = token.split("\\.");
        String claims = new String(Base64.getUrlDecoder().decode(parts[1]), StandardCharsets.UTF_8);
        String forged = claims.replace("\"role\":\"user\"", "\"role\":\"admin\"");
        String tampered = parts[0] + "."
                + Base64.getUrlEncoder().withoutPadding().encodeToString(forged.getBytes(StandardCharsets.UTF_8))
                + "." + parts[2];
        assertThat(jwt.verify(tampered)).isNull();
    }

    @Test
    void rejectsATokenSignedWithADifferentKeySoAStolenTokenFromAnotherInstallIsUseless() {
        byte[] otherKey = new byte[32];
        Arrays.fill(otherKey, (byte) 9);
        String foreign = new Jwt(purpose -> otherKey).issue("diego", "admin");
        assertThat(jwt.verify(foreign)).isNull();
    }

    @Test
    void rejectsMalformedTokensRatherThanThrowing() {
        assertThat(jwt.verify("not-a-token")).isNull();
        assertThat(jwt.verify("")).isNull();
        assertThat(jwt.verify("a.b.c")).isNull();
    }

    @Test
    void rejectsAnExpiredTokenSoSessionsCannotLiveForever() {
        byte[] key = new byte[32];
        Arrays.fill(key, (byte) 3);
        String alreadyExpired = new Jwt(purpose -> key, -60).issue("diego", "admin");
        assertThat(jwt.verify(alreadyExpired)).isNull();
    }

    @Test
    void acceptsATokenThatHasNotExpiredYet() {
        byte[] key = new byte[32];
        Arrays.fill(key, (byte) 3);
        String fresh = new Jwt(purpose -> key, 60).issue("diego", "admin");
        assertThat(jwt.verify(fresh)).isNotNull();
    }
}
