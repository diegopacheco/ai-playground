package com.taxservice.domain;

public record TaxReturn(
        FilingStatus filingStatus,
        long grossIncome,
        int dependents,
        long itemizedDeductions) {
}
