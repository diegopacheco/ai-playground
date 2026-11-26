package com.github.diegopacheco.decisions.judge;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/judge")
public class JudgeController {

    private final JudgeService service;

    public JudgeController(JudgeService service) {
        this.service = service;
    }

    @PostMapping
    public JudgeResult judge(@Valid @RequestBody JudgeRequest request) {
        return service.judge(request.question(), request.answer());
    }

    public record JudgeRequest(@NotBlank String question, @NotBlank String answer) {
    }
}
