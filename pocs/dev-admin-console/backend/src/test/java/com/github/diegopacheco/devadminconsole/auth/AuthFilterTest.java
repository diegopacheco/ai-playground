package com.github.diegopacheco.devadminconsole.auth;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AuthFilterTest {
    private AuthFilter filter;

    @BeforeEach
    void setUp() {
        byte[] key = new byte[32];
        Arrays.fill(key, (byte) 5);
        filter = new AuthFilter(new Jwt(purpose -> key));
    }

    @Test
    void leavesLoginReachableSoNobodyCanEverBeLockedOut() {
        assertThat(filter.protectedPath("/api/auth/login")).isFalse();
    }

    @Test
    void leavesHealthReachableSoStartupProbesDoNotNeedCredentials() {
        assertThat(filter.protectedPath("/actuator/health")).isFalse();
    }

    @Test
    void protectsEveryApiRouteAndSwaggerSoTheConsoleIsNeverAnonymouslyBrowsable() {
        assertThat(filter.protectedPath("/api/projects")).isTrue();
        assertThat(filter.protectedPath("/api/connections/1/query")).isTrue();
        assertThat(filter.protectedPath("/swagger")).isTrue();
        assertThat(filter.protectedPath("/swagger-ui/index.html")).isTrue();
        assertThat(filter.protectedPath("/v3/api-docs")).isTrue();
    }

    @Test
    void restrictsUserManagementAndAuditToAdminsBecauseTheyExposeEveryonesActivity() {
        assertThat(filter.requiresAdmin("/api/users", "GET")).isTrue();
        assertThat(filter.requiresAdmin("/api/audit", "GET")).isTrue();
        assertThat(filter.requiresAdmin("/api/audit/export.csv", "GET")).isTrue();
    }

    @Test
    void restrictsConfigChangesToAdminsWhileLettingAnyUserReadTheProjectList() {
        assertThat(filter.requiresAdmin("/api/projects", "GET")).isFalse();
        assertThat(filter.requiresAdmin("/api/projects", "POST")).isTrue();
        assertThat(filter.requiresAdmin("/api/projects/1", "DELETE")).isTrue();
        assertThat(filter.requiresAdmin("/api/projects/1/connections/2", "PUT")).isTrue();
    }

    @Test
    void letsAnyUserManageSavedQueriesBecauseTheLibraryIsSharedNotAdministrative() {
        assertThat(filter.requiresAdmin("/api/projects/1/saved", "POST")).isFalse();
        assertThat(filter.requiresAdmin("/api/projects/1/saved/2", "DELETE")).isFalse();
    }

    @Test
    void letsAnyUserRunQueriesSoTheConsoleIsUsableWithoutAdminRights() {
        assertThat(filter.requiresAdmin("/api/connections/1/query", "POST")).isFalse();
        assertThat(filter.requiresAdmin("/api/connections/1/schema", "GET")).isFalse();
        assertThat(filter.requiresAdmin("/api/history", "GET")).isFalse();
    }
}
