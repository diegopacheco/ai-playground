package com.github.controlpanel.repo;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Column;
import org.springframework.data.relational.core.mapping.Table;

import java.time.LocalDateTime;

@Table("repository")
public record RepositoryRow(
        @Id Long id,
        String owner,
        String name,
        @Column("full_name") String fullName,
        @Column("added_at") LocalDateTime addedAt,
        @Column("last_synced_at") LocalDateTime lastSyncedAt
) {
}
