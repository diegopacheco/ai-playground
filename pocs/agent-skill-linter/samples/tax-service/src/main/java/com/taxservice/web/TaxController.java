package com.taxservice.web;

import com.taxservice.domain.TaxResult;
import com.taxservice.domain.TaxReturn;
import com.taxservice.service.TaxCalculator;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/tax")
public class TaxController {

    private final TaxCalculator taxCalculator;

    public TaxController(TaxCalculator taxCalculator) {
        this.taxCalculator = taxCalculator;
    }

    @GetMapping("/health")
    public String health() {
        return "ok";
    }

    @PostMapping("/calculate")
    public TaxResult calculate(@RequestBody TaxReturn taxReturn) {
        return taxCalculator.calculate(taxReturn);
    }
}
