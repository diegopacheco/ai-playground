package com.github.controlpanel.repo;

import org.springframework.data.repository.CrudRepository;

import java.util.Optional;

public interface RepositoryRepository extends CrudRepository<RepositoryRow, Long> {
    Optional<RepositoryRow> findByFullName(String fullName);
}
