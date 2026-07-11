package com.diegopacheco.temporalpoc.domain;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

import java.time.OffsetDateTime;

@Table("research_report")
public record ResearchReport(
        @Id Long id,
        String symbol,
        String company,
        String stockSummary,
        String newsSummary,
        String recommendation,
        int confidence,
        String rationale,
        OffsetDateTime createdAt
) {
}
