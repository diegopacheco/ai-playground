package com.retirement.controller;

import com.retirement.model.RetirementInput;
import com.retirement.model.RetirementResult;
import com.retirement.service.RetirementCalculationService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/retirement")
public class RetirementController {

    private final RetirementCalculationService calculationService;

    public RetirementController(RetirementCalculationService calculationService) {
        this.calculationService = calculationService;
    }

    @PostMapping("/calculate")
    public ResponseEntity<?> calculate(@Valid @RequestBody RetirementInput input) {
        try {
            RetirementResult result = calculationService.calculate(input);
            return ResponseEntity.ok(result);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "UP"));
    }
}
