package com.taxservice.service;

import com.taxservice.domain.FilingStatus;
import com.taxservice.domain.TaxBracket;
import com.taxservice.domain.TaxResult;
import com.taxservice.domain.TaxReturn;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class TaxCalculator {

    private final DeductionRules deductionRules;

    public TaxCalculator(DeductionRules deductionRules) {
        this.deductionRules = deductionRules;
    }

    public TaxResult calculate(TaxReturn taxReturn) {
        if (taxReturn.grossIncome() < 0) {
            throw new IllegalArgumentException("grossIncome cannot be negative");
        }
        if (taxReturn.dependents() < 0) {
            throw new IllegalArgumentException("dependents cannot be negative");
        }

        long deduction = deductionRules.applicableDeduction(
                taxReturn.filingStatus(), taxReturn.itemizedDeductions());
        long taxableIncome = Math.max(0, taxReturn.grossIncome() - deduction);

        List<TaxBracket> brackets = bracketsFor(taxReturn.filingStatus());
        long taxBeforeCredits = 0;
        double marginalRate = 0;
        List<String> notes = new ArrayList<>();

        for (TaxBracket bracket : brackets) {
            if (taxableIncome > bracket.lowerBound()) {
                long upper = Math.min(taxableIncome, bracket.upperBound());
                long span = upper - bracket.lowerBound();
                if (span > 0) {
                    long bracketTax = Math.round(span * bracket.rate());
                    taxBeforeCredits += bracketTax;
                    marginalRate = bracket.rate();
                    notes.add("rate " + bracket.rate() + " applied to " + span);
                }
            }
        }

        long credits = childTaxCredit(taxReturn, taxableIncome);
        long taxOwed = Math.max(0, taxBeforeCredits - credits);
        double effectiveRate = taxReturn.grossIncome() == 0
                ? 0.0
                : (double) taxOwed / taxReturn.grossIncome();

        return new TaxResult(taxableIncome, deduction, taxBeforeCredits,
                credits, taxOwed, effectiveRate, marginalRate, notes);
    }

    private long childTaxCredit(TaxReturn taxReturn, long taxableIncome) {
        if (taxReturn.dependents() == 0) {
            return 0;
        }
        long baseCredit = (long) taxReturn.dependents() * 2000;
        long phaseOutThreshold = switch (taxReturn.filingStatus()) {
            case MARRIED_FILING_JOINTLY -> 400000;
            default -> 200000;
        };
        if (taxableIncome <= phaseOutThreshold) {
            return baseCredit;
        }
        long excess = taxableIncome - phaseOutThreshold;
        long steps = (excess + 999) / 1000;
        long reduction = steps * 50;
        if (reduction >= baseCredit) {
            return 0;
        }
        return baseCredit - reduction;
    }

    private List<TaxBracket> bracketsFor(FilingStatus status) {
        return switch (status) {
            case SINGLE -> List.of(
                    new TaxBracket(0.10, 0, 11600),
                    new TaxBracket(0.12, 11600, 47150),
                    new TaxBracket(0.22, 47150, 100525),
                    new TaxBracket(0.24, 100525, 191950),
                    new TaxBracket(0.32, 191950, 243725),
                    new TaxBracket(0.35, 243725, 609350),
                    new TaxBracket(0.37, 609350, Long.MAX_VALUE));
            case MARRIED_FILING_JOINTLY -> List.of(
                    new TaxBracket(0.10, 0, 23200),
                    new TaxBracket(0.12, 23200, 94300),
                    new TaxBracket(0.22, 94300, 201050),
                    new TaxBracket(0.24, 201050, 383900),
                    new TaxBracket(0.32, 383900, 487450),
                    new TaxBracket(0.35, 487450, 731200),
                    new TaxBracket(0.37, 731200, Long.MAX_VALUE));
            case HEAD_OF_HOUSEHOLD -> List.of(
                    new TaxBracket(0.10, 0, 16550),
                    new TaxBracket(0.12, 16550, 63100),
                    new TaxBracket(0.22, 63100, 100500),
                    new TaxBracket(0.24, 100500, 191950),
                    new TaxBracket(0.32, 191950, 243700),
                    new TaxBracket(0.35, 243700, 609350),
                    new TaxBracket(0.37, 609350, Long.MAX_VALUE));
        };
    }
}
