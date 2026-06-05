package com.github.controlpanel.github;

import com.github.controlpanel.common.Label;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Component
public class GithubClient {

    private static final ParameterizedTypeReference<List<Map<String, Object>>> LIST_OF_MAPS =
            new ParameterizedTypeReference<>() {
            };

    private final RestClient client;
    private final int prPageSize;
    private final int issuePageSize;

    public GithubClient(
            @Value("${github.api-url}") String apiUrl,
            @Value("${github.pr-page-size}") int prPageSize,
            @Value("${github.issue-page-size}") int issuePageSize) {
        this.client = RestClient.builder()
                .baseUrl(apiUrl)
                .defaultHeader(HttpHeaders.USER_AGENT, "github-control-panel")
                .defaultHeader(HttpHeaders.ACCEPT, "application/vnd.github+json")
                .defaultHeader("X-GitHub-Api-Version", "2022-11-28")
                .build();
        this.prPageSize = prPageSize;
        this.issuePageSize = issuePageSize;
    }

    public RepoData.Repo fetch(String owner, String name, String token) {
        List<Map<String, Object>> pulls;
        try {
            pulls = getList(token, "/repos/{owner}/{repo}/pulls?state=all&per_page={n}&sort=updated&direction=desc",
                    owner, name, prPageSize);
        } catch (RestClientResponseException ex) {
            if (ex.getStatusCode().value() == 404) {
                return null;
            }
            throw ex;
        }

        List<Map<String, Object>> issues = getList(token,
                "/repos/{owner}/{repo}/issues?state=all&per_page={n}&sort=updated&direction=desc",
                owner, name, issuePageSize);
        List<Map<String, Object>> contributors = getList(token,
                "/repos/{owner}/{repo}/contributors?per_page=100&anon=false", owner, name);

        return new RepoData.Repo(mapPulls(pulls), mapIssues(issues), mapContributors(contributors));
    }

    private List<Map<String, Object>> getList(String token, String uri, Object... vars) {
        List<Map<String, Object>> body = client.get()
                .uri(uri, vars)
                .headers(headers -> {
                    if (token != null && !token.isBlank()) {
                        headers.setBearerAuth(token.trim());
                    }
                })
                .retrieve()
                .body(LIST_OF_MAPS);
        return body == null ? List.of() : body;
    }

    private static List<RepoData.PullData> mapPulls(List<Map<String, Object>> pulls) {
        List<RepoData.PullData> result = new ArrayList<>();
        for (Map<String, Object> pr : pulls) {
            String mergedAt = str(pr, "merged_at");
            String state = mergedAt != null ? "MERGED" : "open".equals(str(pr, "state")) ? "OPEN" : "CLOSED";
            result.add(new RepoData.PullData(
                    integer(pr, "number"),
                    str(pr, "title"),
                    login(pr, "user"),
                    state,
                    bool(pr, "draft"),
                    str(pr, "created_at"),
                    str(pr, "updated_at"),
                    str(pr, "closed_at"),
                    mergedAt,
                    str(pr, "html_url"),
                    listOf(pr, "requested_reviewers").size(),
                    labels(pr)));
        }
        return result;
    }

    private static List<RepoData.IssueData> mapIssues(List<Map<String, Object>> issues) {
        List<RepoData.IssueData> result = new ArrayList<>();
        for (Map<String, Object> issue : issues) {
            if (issue.containsKey("pull_request")) {
                continue;
            }
            List<String> assignees = new ArrayList<>();
            for (Map<String, Object> assignee : listOf(issue, "assignees")) {
                String assigneeLogin = str(assignee, "login");
                if (assigneeLogin != null) {
                    assignees.add(assigneeLogin);
                }
            }
            result.add(new RepoData.IssueData(
                    integer(issue, "number"),
                    str(issue, "title"),
                    login(issue, "user"),
                    "open".equals(str(issue, "state")) ? "OPEN" : "CLOSED",
                    str(issue, "body"),
                    integer(issue, "comments"),
                    assignees.size(),
                    assignees,
                    labels(issue),
                    str(issue, "created_at"),
                    str(issue, "updated_at"),
                    str(issue, "closed_at"),
                    str(issue, "html_url")));
        }
        return result;
    }

    private static List<RepoData.ContributorData> mapContributors(List<Map<String, Object>> contributors) {
        List<RepoData.ContributorData> result = new ArrayList<>();
        for (Map<String, Object> contributor : contributors) {
            String contributorLogin = str(contributor, "login");
            if (contributorLogin != null) {
                result.add(new RepoData.ContributorData(contributorLogin, integer(contributor, "contributions")));
            }
        }
        return result;
    }

    private static List<Label> labels(Map<String, Object> node) {
        List<Label> result = new ArrayList<>();
        for (Map<String, Object> label : listOf(node, "labels")) {
            result.add(new Label(str(label, "name"), str(label, "color")));
        }
        return result;
    }

    private static String login(Map<String, Object> node, String key) {
        Object value = node.get(key);
        if (value instanceof Map<?, ?> user) {
            Object loginValue = user.get("login");
            return loginValue == null ? null : loginValue.toString();
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> listOf(Map<String, Object> node, String key) {
        Object value = node.get(key);
        return value instanceof List<?> list ? (List<Map<String, Object>>) list : List.of();
    }

    private static String str(Map<String, Object> node, String key) {
        Object value = node.get(key);
        return value == null ? null : value.toString();
    }

    private static int integer(Map<String, Object> node, String key) {
        Object value = node.get(key);
        return value instanceof Number number ? number.intValue() : 0;
    }

    private static boolean bool(Map<String, Object> node, String key) {
        Object value = node.get(key);
        return value instanceof Boolean flag && flag;
    }
}
