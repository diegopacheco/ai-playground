package com.taxservice;

import com.taxservice.domain.FilingStatus;
import com.taxservice.domain.TaxResult;
import com.taxservice.domain.TaxReturn;
import com.taxservice.service.DeductionRules;
import com.taxservice.service.TaxCalculator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TaxCalculatorTest {

    private TaxCalculator calculator;

    @BeforeEach
    void setUp() {
        calculator = new TaxCalculator(new DeductionRules());
    }

    @Test
    void appliesStandardDeductionForSingleFiler() {
        TaxReturn input = new TaxReturn(FilingStatus.SINGLE, 50000, 0, 0);
        TaxResult result = calculator.calculate(input);
        assertEquals(35400, result.taxableIncome());
    }

    @Test
    void computesProgressiveTaxForSingleFiler() {
        TaxReturn input = new TaxReturn(FilingStatus.SINGLE, 50000, 0, 0);
        TaxResult result = calculator.calculate(input);
        assertTrue(result.taxOwed() > 0);
        assertTrue(result.effectiveRate() > 0 && result.effectiveRate() < 0.22);
    }

    @Test
    void usesItemizedWhenLargerThanStandard() {
        TaxReturn input = new TaxReturn(FilingStatus.SINGLE, 80000, 0, 30000);
        TaxResult result = calculator.calculate(input);
        assertEquals(50000, result.taxableIncome());
        assertEquals(30000, result.deductionApplied());
    }

    @Test
    void appliesChildTaxCreditWithoutPhaseOut() {
        TaxReturn input = new TaxReturn(FilingStatus.MARRIED_FILING_JOINTLY, 90000, 2, 0);
        TaxResult result = calculator.calculate(input);
        assertEquals(4000, result.credits());
    }

    @Test
    void rejectsNegativeIncome() {
        TaxReturn input = new TaxReturn(FilingStatus.SINGLE, -1, 0, 0);
        assertThrows(IllegalArgumentException.class, () -> calculator.calculate(input));
    }

    @Test
    void zeroIncomeHasZeroEffectiveRate() {
        TaxReturn input = new TaxReturn(FilingStatus.SINGLE, 0, 0, 0);
        TaxResult result = calculator.calculate(input);
        assertEquals(0.0, result.effectiveRate());
        assertEquals(0, result.taxOwed());
    }
}
