package com.loan.service;

import com.loan.domain.AutoLoanDecision;
import com.loan.domain.AutoLoanInput;
import org.springframework.stereotype.Service;

@Service
public class LoanService {

    private static final double MAX_LTV = 0.85;
    private static final double MAX_DTI = 0.40;

    public AutoLoanDecision evaluate(AutoLoanInput input) {
        double rate = rateFor(input.creditScore());
        if (rate < 0) {
            return new AutoLoanDecision(false, 0.0, 0.0, "Credit score below minimum (650).");
        }
        if (input.amount() > input.vehicleValue() * MAX_LTV) {
            return new AutoLoanDecision(false, 0.0, rate,
                    "Loan amount exceeds 85% of vehicle value.");
        }
        double monthly = monthlyPayment(input.amount(), rate, input.termMonths());
        double monthlyIncome = input.annualIncome() / 12.0;
        if (monthly > monthlyIncome * MAX_DTI) {
            return new AutoLoanDecision(false, round(monthly), rate,
                    "Monthly payment exceeds 40% of monthly income.");
        }
        return new AutoLoanDecision(true, round(monthly), rate, "Approved.");
    }

    private double rateFor(int creditScore) {
        if (creditScore >= 800) return 5.0;
        if (creditScore >= 700) return 7.0;
        if (creditScore >= 650) return 10.0;
        return -1.0;
    }

    private double monthlyPayment(double principal, double annualRatePct, int months) {
        double r = (annualRatePct / 100.0) / 12.0;
        if (r == 0.0) return principal / months;
        return principal * (r * Math.pow(1 + r, months)) / (Math.pow(1 + r, months) - 1);
    }

    private double round(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
