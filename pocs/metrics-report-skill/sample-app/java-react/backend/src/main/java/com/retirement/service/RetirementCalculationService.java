package com.retirement.service;

import com.retirement.model.RetirementInput;
import com.retirement.model.RetirementResult;
import com.retirement.model.YearlyProjection;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class RetirementCalculationService {

    public void validateBusinessRules(RetirementInput input) {
        if (input.getRetirementAge() <= input.getCurrentAge()) {
            throw new IllegalArgumentException("Retirement age must be greater than current age");
        }
        if (input.getRetirementAge() - input.getCurrentAge() < 5) {
            throw new IllegalArgumentException("Must have at least 5 years until retirement");
        }
        if (input.getLifeExpectancy() <= input.getRetirementAge()) {
            throw new IllegalArgumentException("Life expectancy must be greater than retirement age");
        }
        if (input.getExpectedAnnualReturn() <= input.getInflationRate()) {
            throw new IllegalArgumentException("Expected return should be greater than inflation rate for real growth");
        }
    }

    public RetirementResult calculate(RetirementInput input) {
        validateBusinessRules(input);

        RetirementResult result = new RetirementResult();
        int yearsToRetirement = input.getRetirementAge() - input.getCurrentAge();
        int yearsInRetirement = input.getLifeExpectancy() - input.getRetirementAge();
        double monthlyRate = input.getExpectedAnnualReturn() / 100.0 / 12.0;
        double annualRate = input.getExpectedAnnualReturn() / 100.0;
        double inflationRate = input.getInflationRate() / 100.0;

        List<YearlyProjection> projections = new ArrayList<>();
        double balance = input.getCurrentSavings();
        double totalContributions = input.getCurrentSavings();

        for (int year = 1; year <= yearsToRetirement; year++) {
            double startBalance = balance;
            double yearlyContribution = input.getMonthlyContribution() * 12;
            double interestEarned = 0;

            for (int month = 0; month < 12; month++) {
                balance += input.getMonthlyContribution();
                interestEarned += balance * monthlyRate;
                balance += balance * monthlyRate;
            }

            totalContributions += yearlyContribution;
            projections.add(new YearlyProjection(
                    year,
                    input.getCurrentAge() + year,
                    Math.round(startBalance * 100.0) / 100.0,
                    Math.round(yearlyContribution * 100.0) / 100.0,
                    Math.round(interestEarned * 100.0) / 100.0,
                    Math.round(balance * 100.0) / 100.0
            ));
        }

        double totalSavingsAtRetirement = balance;
        double inflationFactor = Math.pow(1 + inflationRate, yearsToRetirement);
        double inflationAdjustedMonthlyIncome = input.getDesiredMonthlyIncome() * inflationFactor;
        double totalNeeded = inflationAdjustedMonthlyIncome * 12 * yearsInRetirement;

        double withdrawalRate = annualRate > 0 ? annualRate * 0.6 : 0.04;
        double monthlyIncomeFromSavings = (totalSavingsAtRetirement * withdrawalRate) / 12.0;

        double savingsGap = totalNeeded - totalSavingsAtRetirement;

        double requiredMonthlySavings = calculateRequiredMonthlySavings(
                totalNeeded, input.getCurrentSavings(), monthlyRate, yearsToRetirement * 12);

        String riskAssessment = assessRisk(input, totalSavingsAtRetirement, totalNeeded);
        List<String> recommendations = generateRecommendations(input, totalSavingsAtRetirement, totalNeeded, savingsGap);

        result.setTotalSavingsAtRetirement(Math.round(totalSavingsAtRetirement * 100.0) / 100.0);
        result.setTotalNeededForRetirement(Math.round(totalNeeded * 100.0) / 100.0);
        result.setMonthlyIncomeFromSavings(Math.round(monthlyIncomeFromSavings * 100.0) / 100.0);
        result.setSavingsGap(Math.round(savingsGap * 100.0) / 100.0);
        result.setOnTrack(totalSavingsAtRetirement >= totalNeeded);
        result.setYearsToRetirement(yearsToRetirement);
        result.setYearsInRetirement(yearsInRetirement);
        result.setTotalContributions(Math.round(totalContributions * 100.0) / 100.0);
        result.setTotalInterestEarned(Math.round((totalSavingsAtRetirement - totalContributions) * 100.0) / 100.0);
        result.setInflationAdjustedMonthlyIncome(Math.round(inflationAdjustedMonthlyIncome * 100.0) / 100.0);
        result.setRequiredMonthlySavings(Math.round(requiredMonthlySavings * 100.0) / 100.0);
        result.setRiskAssessment(riskAssessment);
        result.setYearlyProjections(projections);
        result.setRecommendations(recommendations);

        return result;
    }

    private double calculateRequiredMonthlySavings(double targetAmount, double currentSavings, double monthlyRate, int months) {
        if (monthlyRate == 0) {
            return (targetAmount - currentSavings) / months;
        }
        double futureValueOfCurrent = currentSavings * Math.pow(1 + monthlyRate, months);
        double remaining = targetAmount - futureValueOfCurrent;
        if (remaining <= 0) return 0;
        return remaining * monthlyRate / (Math.pow(1 + monthlyRate, months) - 1);
    }

    private String assessRisk(RetirementInput input, double projected, double needed) {
        double ratio = projected / needed;
        if (ratio >= 1.5) return "LOW";
        if (ratio >= 1.0) return "MODERATE";
        if (ratio >= 0.7) return "HIGH";
        return "CRITICAL";
    }

    private List<String> generateRecommendations(RetirementInput input, double projected, double needed, double gap) {
        List<String> recs = new ArrayList<>();
        double ratio = projected / needed;

        if (ratio < 1.0) {
            double additionalMonthly = gap / ((input.getRetirementAge() - input.getCurrentAge()) * 12);
            recs.add(String.format("Increase monthly contributions by $%.2f to close the gap", additionalMonthly));
        }

        if (input.getExpectedAnnualReturn() < 6) {
            recs.add("Consider diversifying into higher-return investments like index funds");
        }

        if (input.getRetirementAge() < 65) {
            recs.add("Delaying retirement by a few years can significantly improve your outlook");
        }

        if (input.getMonthlyContribution() < input.getDesiredMonthlyIncome() * 0.15) {
            recs.add("Your savings rate is low relative to your income goal - aim to save at least 15% of desired income");
        }

        if (ratio >= 1.0 && ratio < 1.3) {
            recs.add("You are on track but with a thin margin - consider building a larger buffer");
        }

        if (ratio >= 1.5) {
            recs.add("You are well ahead of your retirement goal - consider early retirement or lifestyle upgrades");
        }

        if (input.getInflationRate() > 4) {
            recs.add("High inflation assumption - consider inflation-protected securities (TIPS)");
        }

        if (input.getLifeExpectancy() - input.getRetirementAge() > 30) {
            recs.add("Long retirement horizon - ensure portfolio has growth allocation for longevity risk");
        }

        if (recs.isEmpty()) {
            recs.add("Your retirement plan looks solid - maintain current contributions and review annually");
        }

        return recs;
    }
}
