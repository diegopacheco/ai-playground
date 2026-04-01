use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetirementInput {
    pub current_age: u32,
    pub retirement_age: u32,
    pub current_savings: f64,
    pub monthly_contribution: f64,
    pub expected_annual_return: f64,
    pub desired_monthly_income: f64,
    pub life_expectancy: u32,
    pub inflation_rate: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetirementResult {
    pub total_savings_at_retirement: f64,
    pub total_needed_for_retirement: f64,
    pub monthly_income_from_savings: f64,
    pub savings_gap: f64,
    pub on_track: bool,
    pub years_to_retirement: u32,
    pub years_in_retirement: u32,
    pub total_contributions: f64,
    pub total_interest_earned: f64,
    pub inflation_adjusted_monthly_income: f64,
    pub required_monthly_savings: f64,
    pub risk_assessment: String,
    pub yearly_projections: Vec<YearlyProjection>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct YearlyProjection {
    pub year: u32,
    pub age: u32,
    pub start_balance: f64,
    pub contributions: f64,
    pub interest_earned: f64,
    pub end_balance: f64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
}
