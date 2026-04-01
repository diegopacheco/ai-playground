package com.retirement.service;

import com.retirement.model.RetirementInput;
import com.retirement.model.RetirementResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RetirementCalculationServiceTest {

    private RetirementCalculationService service;

    @BeforeEach
    void setUp() {
        service = new RetirementCalculationService();
    }

    private RetirementInput createValidInput() {
        RetirementInput input = new RetirementInput();
        input.setCurrentAge(30);
        input.setRetirementAge(65);
        input.setCurrentSavings(50000.0);
        input.setMonthlyContribution(1000.0);
        input.setExpectedAnnualReturn(7.0);
        input.setDesiredMonthlyIncome(5000.0);
        input.setLifeExpectancy(90);
        input.setInflationRate(3.0);
        return input;
    }

    @Test
    void testValidCalculationReturnsResult() {
        RetirementResult result = service.calculate(createValidInput());
        assertNotNull(result);
        assertTrue(result.getTotalSavingsAtRetirement() > 0);
    }

    @Test
    void testYearsToRetirementCalculation() {
        RetirementResult result = service.calculate(createValidInput());
        assertEquals(35, result.getYearsToRetirement());
    }

    @Test
    void testYearsInRetirementCalculation() {
        RetirementResult result = service.calculate(createValidInput());
        assertEquals(25, result.getYearsInRetirement());
    }

    @Test
    void testRetirementAgeNotGreaterThanCurrentAge() {
        RetirementInput input = createValidInput();
        input.setRetirementAge(25);
        assertThrows(IllegalArgumentException.class, () -> service.calculate(input));
    }

    @Test
    void testLessThanFiveYearsToRetirement() {
        RetirementInput input = createValidInput();
        input.setCurrentAge(62);
        input.setRetirementAge(65);
        assertThrows(IllegalArgumentException.class, () -> service.calculate(input));
    }

    @Test
    void testLifeExpectancyLessThanRetirementAge() {
        RetirementInput input = createValidInput();
        input.setLifeExpectancy(60);
        assertThrows(IllegalArgumentException.class, () -> service.calculate(input));
    }

    @Test
    void testReturnLessThanInflation() {
        RetirementInput input = createValidInput();
        input.setExpectedAnnualReturn(2.0);
        input.setInflationRate(5.0);
        assertThrows(IllegalArgumentException.class, () -> service.calculate(input));
    }

    @Test
    void testProjectionsCountMatchesYears() {
        RetirementResult result = service.calculate(createValidInput());
        assertEquals(35, result.getYearlyProjections().size());
    }

    @Test
    void testTotalContributionsIncludeInitialSavings() {
        RetirementResult result = service.calculate(createValidInput());
        double expectedContributions = 50000.0 + (1000.0 * 12 * 35);
        assertEquals(expectedContributions, result.getTotalContributions(), 0.01);
    }

    @Test
    void testRecommendationsNotEmpty() {
        RetirementResult result = service.calculate(createValidInput());
        assertNotNull(result.getRecommendations());
        assertFalse(result.getRecommendations().isEmpty());
    }
}
