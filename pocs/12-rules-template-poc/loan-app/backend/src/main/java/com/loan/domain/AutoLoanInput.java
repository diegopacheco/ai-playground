package com.loan.domain;

public record AutoLoanInput(
        double amount,
        int termMonths,
        double annualIncome,
        double vehicleValue,
        int creditScore
) {}
