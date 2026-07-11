package com.diegopacheco.temporalpoc.api;

import com.diegopacheco.temporalpoc.domain.ResearchReport;

import java.util.List;

public record ResearchPage(
        List<ResearchReport> content,
        int page,
        int size,
        long totalElements,
        int totalPages
) {
}
