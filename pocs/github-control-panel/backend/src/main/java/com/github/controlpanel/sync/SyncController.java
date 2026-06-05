package com.github.controlpanel.sync;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/sync")
public class SyncController {

    private final SyncService service;

    public SyncController(SyncService service) {
        this.service = service;
    }

    public record SyncResponse(int repos, List<SyncService.RepoResult> results) {
    }

    @PostMapping
    public SyncResponse sync(@RequestHeader(value = "X-GitHub-Token", required = false) String token) {
        List<SyncService.RepoResult> results = service.sync(token);
        return new SyncResponse(results.size(), results);
    }
}
