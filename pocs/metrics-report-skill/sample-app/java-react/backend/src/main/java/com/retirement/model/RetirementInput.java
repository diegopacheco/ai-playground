package com.retirement.model;

import jakarta.validation.constraints.*;

public class RetirementInput {

    @NotNull(message = "Current age is required")
    @Min(value = 18, message = "Age must be at least 18")
    @Max(value = 80, message = "Age must be at most 80")
    private Integer currentAge;

    @NotNull(message = "Retirement age is required")
    @Min(value = 50, message = "Retirement age must be at least 50")
    @Max(value = 75, message = "Retirement age must be at most 75")
    private Integer retirementAge;

    @NotNull(message = "Current savings is required")
    @Min(value = 0, message = "Current savings cannot be negative")
    private Double currentSavings;

    @NotNull(message = "Monthly contribution is required")
    @Min(value = 0, message = "Monthly contribution cannot be negative")
    @Max(value = 100000, message = "Monthly contribution cannot exceed 100,000")
    private Double monthlyContribution;

    @NotNull(message = "Expected annual return is required")
    @Min(value = 0, message = "Expected return cannot be negative")
    @Max(value = 30, message = "Expected return cannot exceed 30%")
    private Double expectedAnnualReturn;

    @NotNull(message = "Desired monthly income is required")
    @Min(value = 500, message = "Desired monthly income must be at least 500")
    private Double desiredMonthlyIncome;

    @NotNull(message = "Life expectancy is required")
    @Min(value = 70, message = "Life expectancy must be at least 70")
    @Max(value = 110, message = "Life expectancy must be at most 110")
    private Integer lifeExpectancy;

    @NotNull(message = "Inflation rate is required")
    @Min(value = 0, message = "Inflation rate cannot be negative")
    @Max(value = 15, message = "Inflation rate cannot exceed 15%")
    private Double inflationRate;

    public Integer getCurrentAge() { return currentAge; }
    public void setCurrentAge(Integer currentAge) { this.currentAge = currentAge; }
    public Integer getRetirementAge() { return retirementAge; }
    public void setRetirementAge(Integer retirementAge) { this.retirementAge = retirementAge; }
    public Double getCurrentSavings() { return currentSavings; }
    public void setCurrentSavings(Double currentSavings) { this.currentSavings = currentSavings; }
    public Double getMonthlyContribution() { return monthlyContribution; }
    public void setMonthlyContribution(Double monthlyContribution) { this.monthlyContribution = monthlyContribution; }
    public Double getExpectedAnnualReturn() { return expectedAnnualReturn; }
    public void setExpectedAnnualReturn(Double expectedAnnualReturn) { this.expectedAnnualReturn = expectedAnnualReturn; }
    public Double getDesiredMonthlyIncome() { return desiredMonthlyIncome; }
    public void setDesiredMonthlyIncome(Double desiredMonthlyIncome) { this.desiredMonthlyIncome = desiredMonthlyIncome; }
    public Integer getLifeExpectancy() { return lifeExpectancy; }
    public void setLifeExpectancy(Integer lifeExpectancy) { this.lifeExpectancy = lifeExpectancy; }
    public Double getInflationRate() { return inflationRate; }
    public void setInflationRate(Double inflationRate) { this.inflationRate = inflationRate; }
}
