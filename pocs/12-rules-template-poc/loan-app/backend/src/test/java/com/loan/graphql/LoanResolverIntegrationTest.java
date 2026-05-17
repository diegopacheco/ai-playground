package com.loan.graphql;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class LoanResolverIntegrationTest {

    @LocalServerPort
    int port;

    private final HttpClient http = HttpClient.newHttpClient();

    @Test
    void healthQueryReturnsOk() throws Exception {
        String body = postGraphql("{\"query\":\"{ health }\"}");
        assertThat(body).contains("\"health\":\"ok\"");
        assertThat(body).doesNotContain("\"errors\"");
    }

    @Test
    void approvedLoanReturnsTermsAndApproval() throws Exception {
        String payload = """
                {"query":"mutation($i:AutoLoanInput!){requestAutoLoan(input:$i){approved monthlyPayment interestRate reason}}",
                 "variables":{"i":{"amount":25000,"termMonths":60,"annualIncome":80000,"vehicleValue":30000,"creditScore":720}}}
                """;
        String body = postGraphql(payload);
        assertThat(body).doesNotContain("\"errors\"");
        assertThat(body).contains("\"approved\":true");
        assertThat(body).contains("\"interestRate\":7.0");
        assertThat(body).contains("\"reason\":\"Approved.\"");
    }

    @Test
    void deniedLoanWhenCreditScoreTooLow() throws Exception {
        String payload = """
                {"query":"mutation($i:AutoLoanInput!){requestAutoLoan(input:$i){approved reason}}",
                 "variables":{"i":{"amount":15000,"termMonths":60,"annualIncome":80000,"vehicleValue":30000,"creditScore":500}}}
                """;
        String body = postGraphql(payload);
        assertThat(body).doesNotContain("\"errors\"");
        assertThat(body).contains("\"approved\":false");
        assertThat(body).contains("Credit score below minimum (650).");
    }

    @Test
    void deniedLoanWhenLoanExceedsLtvCap() throws Exception {
        String payload = """
                {"query":"mutation($i:AutoLoanInput!){requestAutoLoan(input:$i){approved reason}}",
                 "variables":{"i":{"amount":28000,"termMonths":60,"annualIncome":80000,"vehicleValue":30000,"creditScore":720}}}
                """;
        String body = postGraphql(payload);
        assertThat(body).doesNotContain("\"errors\"");
        assertThat(body).contains("\"approved\":false");
        assertThat(body).contains("85% of vehicle value");
    }

    private String postGraphql(String json) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:" + port + "/graphql"))
                .header("content-type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        assertThat(response.statusCode()).isBetween(200, 299);
        return response.body();
    }
}
