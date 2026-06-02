package com.diegopacheco.autotune.config;

import com.diegopacheco.autotune.pattern.BreakerManager;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/config")
public class ConfigController {

    private final BreakerManager manager;

    public ConfigController(BreakerManager manager) {
        this.manager = manager;
    }

    @GetMapping("/circuitbreaker")
    public CircuitBreakerSettings get() {
        return manager.currentSettings();
    }

    @PostMapping("/circuitbreaker")
    public Clamp.Result apply(@RequestBody CircuitBreakerSettings proposed) {
        Clamp.Result clamped = Clamp.apply(proposed);
        manager.apply(clamped.settings());
        return clamped;
    }
}
