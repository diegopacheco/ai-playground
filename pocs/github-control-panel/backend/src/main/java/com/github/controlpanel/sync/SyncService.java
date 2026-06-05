package com.github.controlpanel.sync;

import com.github.controlpanel.github.GithubClient;
import com.github.controlpanel.github.RepoData;
import com.github.controlpanel.repo.RepoService;
import com.github.controlpanel.repo.RepositoryRow;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class SyncService {

    private final RepoService repoService;
    private final GithubClient github;
    private final RepoWriter writer;

    public SyncService(RepoService repoService, GithubClient github, RepoWriter writer) {
        this.repoService = repoService;
        this.github = github;
        this.writer = writer;
    }

    public record RepoResult(String fullName, String status, int pullRequests, int issues, String error) {
    }

    public List<RepoResult> sync(String token) {
        List<RepoResult> results = new ArrayList<>();
        for (RepositoryRow repo : repoService.list()) {
            try {
                RepoData.Repo data = github.fetch(repo.owner(), repo.name(), token);
                if (data == null) {
                    results.add(new RepoResult(repo.fullName(), "not-found", 0, 0, "repository not found"));
                    continue;
                }
                int prs = writer.replace(repo.id(), data);
                results.add(new RepoResult(repo.fullName(), "ok", prs, data.issues().size(), null));
            } catch (Exception ex) {
                results.add(new RepoResult(repo.fullName(), "error", 0, 0, ex.getMessage()));
            }
        }
        return results;
    }
}
