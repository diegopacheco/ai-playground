package com.github.diegopacheco.sandboxspring.controller;

import com.github.diegopacheco.sandboxspring.service.SubagentService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/subagent")
public class SubagentController {

    private final SubagentService subagentService;

    public SubagentController(SubagentService subagentService) {
        this.subagentService = subagentService;
    }

    @PostMapping("/orchestrate")
    public String orchestrate(@RequestBody OrchestrateRequest request) {
        return subagentService.orchestrate(request.task(), request.data());
    }

    record OrchestrateRequest(String task, String data) {}
}
