package com.github.controlpanel.repo;

import com.github.controlpanel.common.Times;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class RepoService {

    private final RepositoryRepository repositories;
    private final JdbcClient jdbc;

    public RepoService(RepositoryRepository repositories, JdbcClient jdbc) {
        this.repositories = repositories;
        this.jdbc = jdbc;
    }

    public List<RepositoryRow> list() {
        List<RepositoryRow> all = new ArrayList<>();
        repositories.findAll().forEach(all::add);
        all.sort((a, b) -> a.fullName().compareToIgnoreCase(b.fullName()));
        return all;
    }

    public List<RepositoryRow> add(List<String> rawRepos) {
        for (String raw : rawRepos) {
            String fullName = normalize(raw);
            if (fullName == null) {
                continue;
            }
            if (repositories.findByFullName(fullName).isPresent()) {
                continue;
            }
            String[] parts = fullName.split("/");
            repositories.save(new RepositoryRow(null, parts[0], parts[1], fullName, Times.now(), null));
        }
        return list();
    }

    public void remove(Long id) {
        jdbc.sql("DELETE FROM pull_request WHERE repository_id = ?").param(id).update();
        jdbc.sql("DELETE FROM issue WHERE repository_id = ?").param(id).update();
        jdbc.sql("DELETE FROM contribution WHERE repository_id = ?").param(id).update();
        repositories.deleteById(id);
    }

    static String normalize(String raw) {
        if (raw == null) {
            return null;
        }
        String value = raw.trim();
        if (value.isEmpty()) {
            return null;
        }
        value = value.replaceFirst("^https?://github.com/", "");
        value = value.replaceFirst("\\.git$", "");
        value = value.replaceAll("/+$", "");
        String[] parts = value.split("/");
        if (parts.length < 2 || parts[0].isBlank() || parts[1].isBlank()) {
            return null;
        }
        return parts[0] + "/" + parts[1];
    }
}
