package com.diegopacheco.autotune.tune;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/tune")
public class TuneController {

    private final OpenAiClient openai;
    private final TuningService tuningService;

    public TuneController(OpenAiClient openai, TuningService tuningService) {
        this.openai = openai;
        this.tuningService = tuningService;
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        return Map.of("configured", openai.configured(), "model", openai.model());
    }

    @PostMapping("/circuitbreaker")
    public ResponseEntity<?> tune(@RequestBody(required = false) RunSummary run) {
        if (!openai.configured()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "OPENAI_API_KEY is not set; cannot call the tuning model."));
        }
        try {
            return ResponseEntity.ok(tuningService.tune(run));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(Map.of("error", e.getMessage()));
        }
    }
}
