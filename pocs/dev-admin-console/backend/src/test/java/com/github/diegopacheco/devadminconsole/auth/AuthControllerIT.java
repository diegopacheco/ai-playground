package com.github.diegopacheco.devadminconsole.auth;

import static org.assertj.core.api.Assertions.assertThat;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration-test")
class AuthControllerIT {
    private static final String BASE = "http://localhost:8099";
    private final HttpClient client = HttpClient.newHttpClient();

    private HttpResponse<String> send(HttpRequest request) throws Exception {
        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    private HttpRequest.Builder request(String path) {
        return HttpRequest.newBuilder(URI.create(BASE + path)).header("Content-Type", "application/json");
    }

    private String login(String username, String password) throws Exception {
        HttpResponse<String> response = send(request("/api/auth/login")
                .POST(HttpRequest.BodyPublishers.ofString("{\"username\":\"" + username + "\",\"password\":\"" + password + "\"}"))
                .build());
        assertThat(response.statusCode()).isEqualTo(200);
        return response.headers().firstValue("set-cookie").orElseThrow().split(";")[0];
    }

    @Test
    void refusesEveryApiRouteWithoutASessionSoTheConsoleIsNeverAnonymouslyReachable() throws Exception {
        assertThat(send(request("/api/projects").GET().build()).statusCode()).isEqualTo(401);
        assertThat(send(request("/api/users").GET().build()).statusCode()).isEqualTo(401);
        assertThat(send(request("/swagger-ui/index.html").GET().build()).statusCode()).isEqualTo(401);
    }

    @Test
    void refusesTheWrongPasswordForTheBootstrapAccount() throws Exception {
        HttpResponse<String> response = send(request("/api/auth/login")
                .POST(HttpRequest.BodyPublishers.ofString("{\"username\":\"admin\",\"password\":\"wrong\"}")).build());
        assertThat(response.statusCode()).isEqualTo(401);
    }

    @Test
    void issuesASessionForTheBootstrapAdminAndReportsTheDefaultPasswordIsStillInUse() throws Exception {
        String cookie = login("admin", "admin");
        HttpResponse<String> session = send(request("/api/auth/session").header("Cookie", cookie).GET().build());
        assertThat(session.body()).contains("\"role\":\"admin\"").contains("\"usingBootstrapPassword\":true");
    }

    @Test
    void neverTellsANonAdminThatTheAdminAccountStillUsesItsDefaultPassword() throws Exception {
        String adminCookie = login("admin", "admin");
        send(request("/api/users").header("Cookie", adminCookie)
                .POST(HttpRequest.BodyPublishers.ofString("{\"username\":\"probe\",\"password\":\"probe\",\"role\":\"user\"}")).build());
        String readerCookie = login("probe", "probe");
        HttpResponse<String> session = send(request("/api/auth/session").header("Cookie", readerCookie).GET().build());
        assertThat(session.body()).contains("\"role\":\"user\"").doesNotContain("usingBootstrapPassword");
    }

    @Test
    void blocksANonAdminFromUserManagementAuditAndConfigChanges() throws Exception {
        String adminCookie = login("admin", "admin");
        send(request("/api/users").header("Cookie", adminCookie)
                .POST(HttpRequest.BodyPublishers.ofString("{\"username\":\"reader2\",\"password\":\"reader2\",\"role\":\"user\"}")).build());
        String cookie = login("reader2", "reader2");
        assertThat(send(request("/api/users").header("Cookie", cookie).GET().build()).statusCode()).isEqualTo(403);
        assertThat(send(request("/api/audit").header("Cookie", cookie).GET().build()).statusCode()).isEqualTo(403);
        assertThat(send(request("/api/projects").header("Cookie", cookie)
                .POST(HttpRequest.BodyPublishers.ofString("{\"name\":\"nope\"}")).build()).statusCode()).isEqualTo(403);
    }

    @Test
    void refusesToDeleteTheLastAdminThroughTheApiSoTheInstallCannotBeOrphaned() throws Exception {
        String cookie = login("admin", "admin");
        HttpResponse<String> response = send(request("/api/users/1").header("Cookie", cookie).DELETE().build());
        assertThat(response.body()).contains("last admin");
    }
}
