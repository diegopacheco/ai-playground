package com.github.diegopacheco.decisions.guardrail;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/guardrail")
public class GuardrailController {

    private final GuardrailService service;

    public GuardrailController(GuardrailService service) {
        this.service = service;
    }

    @PostMapping
    public GuardrailResult screen(@Valid @RequestBody GuardrailRequest request) {
        return service.screen(request.battery(), request.text());
    }

    public record GuardrailRequest(@NotNull Battery battery, @NotBlank String text) {
    }
}
