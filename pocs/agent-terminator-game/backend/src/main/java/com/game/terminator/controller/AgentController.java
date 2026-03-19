package com.game.terminator.controller;

import com.game.terminator.agent.AgentRegistry;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/agents")
public class AgentController {

    @GetMapping
    public List<Map<String, Object>> listAgents() {
        return AgentRegistry.toJson();
    }
}
