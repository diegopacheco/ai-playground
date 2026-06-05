package com.github.controlpanel.repo;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/repos")
public class RepoController {

    private final RepoService service;

    public RepoController(RepoService service) {
        this.service = service;
    }

    public record AddReposRequest(List<String> repos) {
    }

    public record RepoView(Long id, String owner, String name, String fullName, String addedAt, String lastSyncedAt) {
    }

    @GetMapping
    public List<RepoView> list() {
        return service.list().stream().map(RepoController::toView).toList();
    }

    @PostMapping
    public List<RepoView> add(@RequestBody AddReposRequest request) {
        List<String> repos = request.repos() == null ? List.of() : request.repos();
        return service.add(repos).stream().map(RepoController::toView).toList();
    }

    @DeleteMapping("/{id}")
    public List<RepoView> remove(@PathVariable Long id) {
        service.remove(id);
        return service.list().stream().map(RepoController::toView).toList();
    }

    private static RepoView toView(RepositoryRow row) {
        return new RepoView(
                row.id(),
                row.owner(),
                row.name(),
                row.fullName(),
                com.github.controlpanel.common.Times.iso(row.addedAt()),
                com.github.controlpanel.common.Times.iso(row.lastSyncedAt())
        );
    }
}
