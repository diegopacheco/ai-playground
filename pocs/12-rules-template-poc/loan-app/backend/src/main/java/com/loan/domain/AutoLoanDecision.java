package com.loan.domain;

public record AutoLoanDecision(
        boolean approved,
        double monthlyPayment,
        double interestRate,
        String reason
) {}
