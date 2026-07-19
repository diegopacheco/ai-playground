package com.github.diegopacheco.devadminconsole.console;

import io.swagger.v3.oas.annotations.Operation;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class RouteController {
    @GetMapping("/swagger")
    @Operation(summary = "Redirect to the Swagger UI")
    public String swagger() {
        return "redirect:/swagger-ui/index.html";
    }
}
