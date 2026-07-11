package com.diegopacheco.temporalpoc.api;

import com.diegopacheco.temporalpoc.domain.ResearchReport;
import com.diegopacheco.temporalpoc.service.ResearchService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/research")
public class ResearchController {
    private static final Logger log = LoggerFactory.getLogger(ResearchController.class);
    private final ResearchService service;

    public ResearchController(ResearchService service) {
        this.service = service;
    }

    @PostMapping
    public ResearchReport research(@RequestBody ResearchRequest request) {
        log.info("blocking research request received symbol={} company={}", request.symbol(), request.company());
        return service.research(request.symbol(), request.company());
    }

    @PostMapping("/trigger")
    public TriggerResponse trigger(@RequestBody ResearchRequest request) {
        log.info("async workflow trigger request received symbol={} company={}", request.symbol(), request.company());
        TriggerResponse response = service.trigger(request.symbol(), request.company());
        log.info("async workflow trigger accepted workflowId={} runId={} temporalUrl={}", response.workflowId(), response.runId(), response.temporalUrl());
        return response;
    }

    @GetMapping
    public ResearchPage reports(@RequestParam(defaultValue = "0") int page, @RequestParam(defaultValue = "10") int size) {
        log.info("research page request received page={} size={}", page, size);
        Page<ResearchReport> reports = service.reports(page, size);
        log.info("research page response page={} size={} totalElements={} totalPages={}", reports.getNumber(), reports.getSize(), reports.getTotalElements(), reports.getTotalPages());
        return new ResearchPage(reports.getContent(), reports.getNumber(), reports.getSize(), reports.getTotalElements(), reports.getTotalPages());
    }
}
