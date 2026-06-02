package com.taxservice;

import com.taxservice.domain.FilingStatus;
import com.taxservice.domain.TaxReturn;
import com.taxservice.service.DeductionRules;
import com.taxservice.service.TaxCalculator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class SlowReconciliationTest {

    @Test
    void fullYearReconciliationAcrossIncomeBands() throws InterruptedException {
        TaxCalculator calculator = new TaxCalculator(new DeductionRules());
        long total = 0;
        for (int income = 10000; income <= 200000; income += 10000) {
            TaxReturn input = new TaxReturn(FilingStatus.SINGLE, income, 1, 0);
            total += calculator.calculate(input).taxOwed();
        }
        Thread.sleep(5200);
        assertTrue(total > 0);
    }
}
