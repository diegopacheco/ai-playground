package com.taxservice.domain;

import java.util.List;

public record TaxResult(
        long taxableIncome,
        long deductionApplied,
        long taxBeforeCredits,
        long credits,
        long taxOwed,
        double effectiveRate,
        double marginalRate,
        List<String> notes) {
}
