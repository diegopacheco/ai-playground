class RetirementCalculationService:

    @staticmethod
    def validate_business_rules(data):
        if data['retirementAge'] <= data['currentAge']:
            raise ValueError("Retirement age must be greater than current age")
        if data['retirementAge'] - data['currentAge'] < 5:
            raise ValueError("At least 5 years gap between current age and retirement age is required")
        if data['lifeExpectancy'] <= data['retirementAge']:
            raise ValueError("Life expectancy must be greater than retirement age")
        if data['expectedAnnualReturn'] <= data['inflationRate']:
            raise ValueError("Expected annual return must be greater than inflation rate")

    @staticmethod
    def calculate(data):
        RetirementCalculationService.validate_business_rules(data)

        current_age = data['currentAge']
        retirement_age = data['retirementAge']
        current_savings = data['currentSavings']
        monthly_contribution = data['monthlyContribution']
        annual_return = data['expectedAnnualReturn']
        desired_monthly_income = data['desiredMonthlyIncome']
        life_expectancy = data['lifeExpectancy']
        inflation_rate = data['inflationRate']

        years_to_retirement = retirement_age - current_age
        years_in_retirement = life_expectancy - retirement_age
        monthly_rate = annual_return / 100 / 12

        balance = current_savings
        total_contributions = current_savings
        total_interest = 0.0
        yearly_projections = []

        for year in range(years_to_retirement):
            start_balance = balance
            yearly_contribution = monthly_contribution * 12
            interest_earned = 0.0

            for _ in range(12):
                balance += monthly_contribution
                interest = balance * monthly_rate
                interest_earned += interest
                balance += interest

            total_contributions += yearly_contribution
            total_interest += interest_earned

            yearly_projections.append({
                'year': year + 1,
                'age': current_age + year + 1,
                'startBalance': round(start_balance, 2),
                'contributions': round(yearly_contribution, 2),
                'interestEarned': round(interest_earned, 2),
                'endBalance': round(balance, 2),
            })

        inflation_factor = (1 + inflation_rate / 100) ** years_to_retirement
        adjusted_monthly_income = desired_monthly_income * inflation_factor
        total_needed = adjusted_monthly_income * 12 * years_in_retirement

        withdrawal_rate = (annual_return / 100) * 0.6
        monthly_income_from_savings = (balance * withdrawal_rate) / 12

        savings_gap = total_needed - balance

        required_monthly = RetirementCalculationService._calculate_required_monthly(
            total_needed, current_savings, monthly_rate, years_to_retirement * 12
        )

        ratio = balance / total_needed if total_needed > 0 else 1.0
        risk_assessment = RetirementCalculationService._assess_risk(ratio)

        recommendations = RetirementCalculationService._generate_recommendations(
            data, balance, total_needed, savings_gap
        )

        return {
            'totalSavingsAtRetirement': round(balance, 2),
            'totalNeededForRetirement': round(total_needed, 2),
            'monthlyIncomeFromSavings': round(monthly_income_from_savings, 2),
            'savingsGap': round(savings_gap, 2),
            'onTrack': balance >= total_needed,
            'yearsToRetirement': years_to_retirement,
            'yearsInRetirement': years_in_retirement,
            'totalContributions': round(total_contributions, 2),
            'totalInterestEarned': round(total_interest, 2),
            'inflationAdjustedMonthlyIncome': round(adjusted_monthly_income, 2),
            'requiredMonthlySavings': round(required_monthly, 2),
            'riskAssessment': risk_assessment,
            'yearlyProjections': yearly_projections,
            'recommendations': recommendations,
        }

    @staticmethod
    def _calculate_required_monthly(target, current_savings, monthly_rate, months):
        if monthly_rate == 0:
            remaining = target - current_savings
            return remaining / months if months > 0 else 0
        future_value_current = current_savings * ((1 + monthly_rate) ** months)
        remaining = target - future_value_current
        if remaining <= 0:
            return 0
        return (remaining * monthly_rate) / (((1 + monthly_rate) ** months) - 1)

    @staticmethod
    def _assess_risk(ratio):
        if ratio >= 1.5:
            return "LOW"
        if ratio >= 1.0:
            return "MODERATE"
        if ratio >= 0.7:
            return "HIGH"
        return "CRITICAL"

    @staticmethod
    def _generate_recommendations(data, projected, needed, gap):
        recommendations = []
        if gap > 0:
            recommendations.append("Consider increasing your monthly contributions to close the savings gap.")
        else:
            recommendations.append("You're on track for your retirement goals. Keep up the great work!")

        if data['expectedAnnualReturn'] < 6:
            recommendations.append("Consider diversifying into higher-return investments to boost growth.")

        if data['retirementAge'] < 65:
            recommendations.append("Delaying retirement by a few years could significantly improve your financial position.")

        if data['monthlyContribution'] < data['desiredMonthlyIncome'] * 0.3:
            recommendations.append("Your savings rate is low relative to your desired retirement income. Try to save at least 30% of your target income.")

        ratio = projected / needed if needed > 0 else 1.0
        if ratio < 0.7:
            recommendations.append("Your retirement readiness is critical. Consider consulting a financial advisor.")

        if data['inflationRate'] > 4:
            recommendations.append("High inflation assumptions are impacting your projections. Consider inflation-protected investments.")

        if data['lifeExpectancy'] - data['retirementAge'] > 30:
            recommendations.append("With a long retirement horizon, ensure your portfolio has growth potential to sustain withdrawals.")

        if not recommendations:
            recommendations.append("Review your retirement plan annually to stay on track.")

        return recommendations
