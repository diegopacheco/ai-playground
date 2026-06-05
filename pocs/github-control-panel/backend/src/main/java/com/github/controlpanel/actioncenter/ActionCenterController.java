package com.github.controlpanel.actioncenter;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/action-center")
public class ActionCenterController {

    private final ActionCenterService service;

    public ActionCenterController(ActionCenterService service) {
        this.service = service;
    }

    @GetMapping
    public ActionCenterService.ActionCenter get() {
        return service.get();
    }
}
