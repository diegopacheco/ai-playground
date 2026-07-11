package com.diegopacheco.temporalpoc.repository;

import com.diegopacheco.temporalpoc.domain.ResearchReport;
import org.springframework.data.repository.PagingAndSortingRepository;
import org.springframework.data.repository.CrudRepository;

public interface ResearchReportRepository extends CrudRepository<ResearchReport, Long>, PagingAndSortingRepository<ResearchReport, Long> {
}
