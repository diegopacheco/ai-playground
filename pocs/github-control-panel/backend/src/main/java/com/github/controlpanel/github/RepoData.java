package com.github.controlpanel.github;

import com.github.controlpanel.common.Label;

import java.util.List;

public final class RepoData {
    private RepoData() {
    }

    public record Repo(List<PullData> pulls, List<IssueData> issues, List<ContributorData> contributors) {
    }

    public record PullData(int number, String title, String author, String state, boolean draft,
                           String createdAt, String updatedAt, String closedAt, String mergedAt,
                           String url, int requestedReviewers, List<Label> labels) {
    }

    public record IssueData(int number, String title, String author, String state, String body,
                            int comments, int assigneesCount, List<String> assignees, List<Label> labels,
                            String createdAt, String updatedAt, String closedAt, String url) {
    }

    public record ContributorData(String login, int contributions) {
    }
}
