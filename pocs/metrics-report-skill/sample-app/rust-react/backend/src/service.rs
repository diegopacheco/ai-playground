use crate::models::{RetirementInput, RetirementResult, YearlyProjection};

pub fn validate(input: &RetirementInput) -> Result<(), String> {
    if input.current_age < 18 || input.current_age > 80 {
        return Err("Current age must be between 18 and 80".into());
    }
    if input.retirement_age < 50 || input.retirement_age > 75 {
        return Err("Retirement age must be between 50 and 75".into());
    }
    if input.retirement_age <= input.current_age {
        return Err("Retirement age must be greater than current age".into());
    }
    if input.retirement_age - input.current_age < 5 {
        return Err("Must have at least 5 years until retirement".into());
    }
    if input.life_expectancy <= input.retirement_age {
        return Err("Life expectancy must be greater than retirement age".into());
    }
    if input.life_expectancy < 70 || input.life_expectancy > 110 {
        return Err("Life expectancy must be between 70 and 110".into());
    }
    if input.expected_annual_return <= input.inflation_rate {
        return Err("Expected return must exceed inflation rate".into());
    }
    if input.monthly_contribution < 0.0 || input.monthly_contribution > 100_000.0 {
        return Err("Monthly contribution must be between 0 and 100,000".into());
    }
    if input.desired_monthly_income < 500.0 {
        return Err("Desired monthly income must be at least 500".into());
    }
    if input.current_savings < 0.0 {
        return Err("Current savings cannot be negative".into());
    }
    if input.expected_annual_return < 0.0 || input.expected_annual_return > 30.0 {
        return Err("Expected annual return must be between 0 and 30".into());
    }
    if input.inflation_rate < 0.0 || input.inflation_rate > 15.0 {
        return Err("Inflation rate must be between 0 and 15".into());
    }
    Ok(())
}

pub fn calculate(input: &RetirementInput) -> RetirementResult {
    let years_to_retirement = input.retirement_age - input.current_age;
    let years_in_retirement = input.life_expectancy - input.retirement_age;
    let monthly_rate = input.expected_annual_return / 100.0 / 12.0;
    let total_months = years_to_retirement * 12;

    let mut balance = input.current_savings;
    let mut total_contributions = input.current_savings;
    let mut projections: Vec<YearlyProjection> = Vec::new();

    for year in 0..years_to_retirement {
        let start_balance = balance;
        let mut yearly_contributions = 0.0;
        let mut yearly_interest = 0.0;

        for _ in 0..12 {
            balance += input.monthly_contribution;
            yearly_contributions += input.monthly_contribution;
            let interest = balance * monthly_rate;
            yearly_interest += interest;
            balance += interest;
        }

        total_contributions += yearly_contributions;

        projections.push(YearlyProjection {
            year: year + 1,
            age: input.current_age + year + 1,
            start_balance: round2(start_balance),
            contributions: round2(yearly_contributions),
            interest_earned: round2(yearly_interest),
            end_balance: round2(balance),
        });
    }

    let total_savings = balance;
    let total_interest = total_savings - total_contributions;

    let inflation_adjusted_monthly =
        input.desired_monthly_income * (1.0 + input.inflation_rate / 100.0).powi(years_to_retirement as i32);
    let total_needed = inflation_adjusted_monthly * 12.0 * years_in_retirement as f64;

    let withdrawal_rate = if input.expected_annual_return > 0.0 {
        (input.expected_annual_return / 100.0) * 0.6
    } else {
        0.04
    };
    let monthly_income_from_savings = total_savings * withdrawal_rate / 12.0;

    let savings_gap = if total_savings < total_needed {
        total_needed - total_savings
    } else {
        0.0
    };
    let on_track = total_savings >= total_needed;

    let required_monthly = calculate_required_monthly(
        total_needed,
        input.current_savings,
        monthly_rate,
        total_months,
    );

    let ratio = if total_needed > 0.0 {
        total_savings / total_needed
    } else {
        2.0
    };
    let risk_assessment = if ratio >= 1.5 {
        "LOW"
    } else if ratio >= 1.0 {
        "MODERATE"
    } else if ratio >= 0.7 {
        "HIGH"
    } else {
        "CRITICAL"
    }
    .to_string();

    let recommendations = generate_recommendations(input, on_track, ratio, &risk_assessment);

    RetirementResult {
        total_savings_at_retirement: round2(total_savings),
        total_needed_for_retirement: round2(total_needed),
        monthly_income_from_savings: round2(monthly_income_from_savings),
        savings_gap: round2(savings_gap),
        on_track,
        years_to_retirement,
        years_in_retirement,
        total_contributions: round2(total_contributions),
        total_interest_earned: round2(total_interest),
        inflation_adjusted_monthly_income: round2(inflation_adjusted_monthly),
        required_monthly_savings: round2(required_monthly),
        risk_assessment,
        yearly_projections: projections,
        recommendations,
    }
}

fn calculate_required_monthly(target: f64, current: f64, monthly_rate: f64, months: u32) -> f64 {
    let n = months as f64;
    let future_current = current * (1.0 + monthly_rate).powf(n);
    let remaining = target - future_current;
    if remaining <= 0.0 {
        return 0.0;
    }
    if monthly_rate == 0.0 {
        return remaining / n;
    }
    let factor = ((1.0 + monthly_rate).powf(n) - 1.0) / monthly_rate;
    remaining / factor
}

fn generate_recommendations(
    input: &RetirementInput,
    on_track: bool,
    ratio: f64,
    risk: &str,
) -> Vec<String> {
    let mut recs = Vec::new();

    if !on_track {
        recs.push(format!(
            "Consider increasing your monthly contribution above ${:.0} to close the savings gap.",
            input.monthly_contribution
        ));
    }

    if input.expected_annual_return < 6.0 {
        recs.push("Consider a more aggressive investment strategy to increase your expected returns.".into());
    }

    if input.monthly_contribution < input.desired_monthly_income * 0.2 {
        recs.push("Your monthly contribution is low relative to your desired retirement income. Try to save at least 20% of your target income.".into());
    }

    if risk == "HIGH" || risk == "CRITICAL" {
        recs.push("Your risk level is elevated. Consider delaying retirement or reducing your desired monthly income.".into());
    }

    if ratio >= 1.5 {
        recs.push("You are well on track. Consider diversifying into lower-risk investments as you approach retirement.".into());
    }

    if input.inflation_rate > 4.0 {
        recs.push("High inflation assumptions significantly impact your retirement needs. Consider inflation-protected securities.".into());
    }

    if on_track && input.retirement_age > 60 {
        recs.push("You may be able to retire earlier than planned given your current savings trajectory.".into());
    }

    if recs.is_empty() {
        recs.push("Keep up your current savings plan and review annually.".into());
    }

    recs
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> RetirementInput {
        RetirementInput {
            current_age: 30,
            retirement_age: 65,
            current_savings: 50000.0,
            monthly_contribution: 1000.0,
            expected_annual_return: 7.0,
            desired_monthly_income: 5000.0,
            life_expectancy: 90,
            inflation_rate: 3.0,
        }
    }

    #[test]
    fn test_valid_calculation_returns_result() {
        let result = calculate(&sample_input());
        assert!(result.total_savings_at_retirement > 0.0);
    }

    #[test]
    fn test_years_to_retirement() {
        let result = calculate(&sample_input());
        assert_eq!(result.years_to_retirement, 35);
    }

    #[test]
    fn test_years_in_retirement() {
        let result = calculate(&sample_input());
        assert_eq!(result.years_in_retirement, 25);
    }

    #[test]
    fn test_retirement_age_not_greater_than_current() {
        let mut input = sample_input();
        input.retirement_age = 25;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn test_less_than_five_years() {
        let mut input = sample_input();
        input.current_age = 62;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn test_life_expectancy_less_than_retirement() {
        let mut input = sample_input();
        input.life_expectancy = 60;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn test_return_less_than_inflation() {
        let mut input = sample_input();
        input.expected_annual_return = 2.0;
        input.inflation_rate = 5.0;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn test_projections_count_matches_years() {
        let result = calculate(&sample_input());
        assert_eq!(result.yearly_projections.len(), 35);
    }

    #[test]
    fn test_total_contributions_include_initial() {
        let result = calculate(&sample_input());
        assert!(result.total_contributions >= 50000.0);
    }

    #[test]
    fn test_recommendations_not_empty() {
        let result = calculate(&sample_input());
        assert!(!result.recommendations.is_empty());
    }
}
