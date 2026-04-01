package com.retirement.model;

import java.util.List;

public class RetirementResult {

    private double totalSavingsAtRetirement;
    private double totalNeededForRetirement;
    private double monthlyIncomeFromSavings;
    private double savingsGap;
    private boolean onTrack;
    private int yearsToRetirement;
    private int yearsInRetirement;
    private double totalContributions;
    private double totalInterestEarned;
    private double inflationAdjustedMonthlyIncome;
    private double requiredMonthlySavings;
    private String riskAssessment;
    private List<YearlyProjection> yearlyProjections;
    private List<String> recommendations;

    public double getTotalSavingsAtRetirement() { return totalSavingsAtRetirement; }
    public void setTotalSavingsAtRetirement(double v) { this.totalSavingsAtRetirement = v; }
    public double getTotalNeededForRetirement() { return totalNeededForRetirement; }
    public void setTotalNeededForRetirement(double v) { this.totalNeededForRetirement = v; }
    public double getMonthlyIncomeFromSavings() { return monthlyIncomeFromSavings; }
    public void setMonthlyIncomeFromSavings(double v) { this.monthlyIncomeFromSavings = v; }
    public double getSavingsGap() { return savingsGap; }
    public void setSavingsGap(double v) { this.savingsGap = v; }
    public boolean isOnTrack() { return onTrack; }
    public void setOnTrack(boolean v) { this.onTrack = v; }
    public int getYearsToRetirement() { return yearsToRetirement; }
    public void setYearsToRetirement(int v) { this.yearsToRetirement = v; }
    public int getYearsInRetirement() { return yearsInRetirement; }
    public void setYearsInRetirement(int v) { this.yearsInRetirement = v; }
    public double getTotalContributions() { return totalContributions; }
    public void setTotalContributions(double v) { this.totalContributions = v; }
    public double getTotalInterestEarned() { return totalInterestEarned; }
    public void setTotalInterestEarned(double v) { this.totalInterestEarned = v; }
    public double getInflationAdjustedMonthlyIncome() { return inflationAdjustedMonthlyIncome; }
    public void setInflationAdjustedMonthlyIncome(double v) { this.inflationAdjustedMonthlyIncome = v; }
    public double getRequiredMonthlySavings() { return requiredMonthlySavings; }
    public void setRequiredMonthlySavings(double v) { this.requiredMonthlySavings = v; }
    public String getRiskAssessment() { return riskAssessment; }
    public void setRiskAssessment(String v) { this.riskAssessment = v; }
    public List<YearlyProjection> getYearlyProjections() { return yearlyProjections; }
    public void setYearlyProjections(List<YearlyProjection> v) { this.yearlyProjections = v; }
    public List<String> getRecommendations() { return recommendations; }
    public void setRecommendations(List<String> v) { this.recommendations = v; }
}
