package com.diegopacheco.autotune.downstream;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/sim")
public class ScenarioController {

    private final ScenarioState scenario;

    public ScenarioController(ScenarioState scenario) {
        this.scenario = scenario;
    }

    @GetMapping("/scenario")
    public Scenario get() {
        return scenario.get();
    }

    @PostMapping("/scenario")
    public Scenario set(@RequestBody Scenario s) {
        scenario.set(s);
        return scenario.get();
    }
}
