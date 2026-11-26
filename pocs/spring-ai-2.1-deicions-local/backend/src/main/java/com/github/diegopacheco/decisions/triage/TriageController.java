package com.github.diegopacheco.decisions.triage;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/triage")
public class TriageController {

    private final TriageService service;

    public TriageController(TriageService service) {
        this.service = service;
    }

    @PostMapping
    public TriageResult triage(@Valid @RequestBody TriageRequest request) {
        return service.triage(request.message());
    }

    public record TriageRequest(@NotBlank String message) {
    }
}
