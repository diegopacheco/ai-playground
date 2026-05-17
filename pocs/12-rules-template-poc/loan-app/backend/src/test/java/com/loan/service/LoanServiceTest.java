package com.loan.service;

import com.loan.domain.AutoLoanDecision;
import com.loan.domain.AutoLoanInput;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class LoanServiceTest {

    private final LoanService service = new LoanService();

    @Test
    void rejectsWhenCreditScoreBelowFloor() {
        AutoLoanDecision d = service.evaluate(input(20000, 60, 80000, 30000, 600));
        assertThat(d.approved()).isFalse();
        assertThat(d.reason()).contains("Credit score below");
        assertThat(d.interestRate()).isZero();
        assertThat(d.monthlyPayment()).isZero();
    }

    @Test
    void rejectsWhenLoanExceedsLtvCap() {
        AutoLoanDecision d = service.evaluate(input(28000, 60, 80000, 30000, 720));
        assertThat(d.approved()).isFalse();
        assertThat(d.reason()).contains("85%");
        assertThat(d.interestRate()).isEqualTo(7.0);
    }

    @Test
    void rejectsWhenMonthlyPaymentExceedsDtiCap() {
        AutoLoanDecision d = service.evaluate(input(25000, 12, 30000, 30000, 660));
        assertThat(d.approved()).isFalse();
        assertThat(d.reason()).contains("40%");
    }

    @Test
    void approvedAtTopTierUsesFivePercent() {
        AutoLoanDecision d = service.evaluate(input(20000, 60, 100000, 30000, 820));
        assertThat(d.approved()).isTrue();
        assertThat(d.interestRate()).isEqualTo(5.0);
    }

    @Test
    void approvedAtMidTierUsesSevenPercentAndMatchesKnownPayment() {
        AutoLoanDecision d = service.evaluate(input(25000, 60, 80000, 30000, 720));
        assertThat(d.approved()).isTrue();
        assertThat(d.interestRate()).isEqualTo(7.0);
        assertThat(d.monthlyPayment()).isCloseTo(495.03, within(0.05));
    }

    @Test
    void approvedAtBottomTierUsesTenPercent() {
        AutoLoanDecision d = service.evaluate(input(15000, 60, 80000, 30000, 660));
        assertThat(d.approved()).isTrue();
        assertThat(d.interestRate()).isEqualTo(10.0);
    }

    @Test
    void monthlyPaymentMatchesStandardAmortizationFormula() {
        AutoLoanDecision d = service.evaluate(input(20000, 60, 100000, 30000, 820));
        assertThat(d.monthlyPayment()).isCloseTo(377.42, within(0.05));
    }

    @Test
    void boundaryCreditScoreSixFiftyIsApproved() {
        AutoLoanDecision d = service.evaluate(input(15000, 60, 80000, 30000, 650));
        assertThat(d.approved()).isTrue();
        assertThat(d.interestRate()).isEqualTo(10.0);
    }

    @Test
    void boundaryCreditScoreSixFortyNineIsRejected() {
        AutoLoanDecision d = service.evaluate(input(15000, 60, 80000, 30000, 649));
        assertThat(d.approved()).isFalse();
    }

    private AutoLoanInput input(double amount, int term, double income, double vehicle, int score) {
        return new AutoLoanInput(amount, term, income, vehicle, score);
    }
}
