package com.taxservice;

import com.taxservice.domain.FilingStatus;
import com.taxservice.service.DeductionRules;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DeductionRulesTest {

    private final DeductionRules rules = new DeductionRules();

    @Test
    void standardDeductionForSingle() {
        assertEquals(14600, rules.standardDeduction(FilingStatus.SINGLE));
    }

    @Test
    void standardDeductionForMarried() {
        assertEquals(29200, rules.standardDeduction(FilingStatus.MARRIED_FILING_JOINTLY));
    }

    @Test
    void itemizedWinsWhenLarger() {
        assertEquals(40000, rules.applicableDeduction(FilingStatus.SINGLE, 40000));
    }

    @Test
    void standardWinsWhenItemizedSmaller() {
        assertEquals(14600, rules.applicableDeduction(FilingStatus.SINGLE, 1000));
    }
}
