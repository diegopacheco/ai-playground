package com.retirement.service

import com.retirement.model.{RetirementInput, RetirementResult, YearlyProjection}
import org.springframework.stereotype.Service
import scala.collection.mutable.ListBuffer

@Service
class RetirementCalculationService:

  def validateBusinessRules(input: RetirementInput): Unit =
    if input.getRetirementAge <= input.getCurrentAge then
      throw IllegalArgumentException("Retirement age must be greater than current age")
    if input.getRetirementAge - input.getCurrentAge < 5 then
      throw IllegalArgumentException("Must have at least 5 years until retirement")
    if input.getLifeExpectancy <= input.getRetirementAge then
      throw IllegalArgumentException("Life expectancy must be greater than retirement age")
    if input.getExpectedAnnualReturn <= input.getInflationRate then
      throw IllegalArgumentException("Expected return should be greater than inflation rate for real growth")

  def calculate(input: RetirementInput): RetirementResult =
    validateBusinessRules(input)

    val yearsToRetirement = input.getRetirementAge - input.getCurrentAge
    val yearsInRetirement = input.getLifeExpectancy - input.getRetirementAge
    val monthlyRate = input.getExpectedAnnualReturn / 100.0 / 12.0
    val annualRate = input.getExpectedAnnualReturn / 100.0
    val inflationRate = input.getInflationRate / 100.0

    val projections = new java.util.ArrayList[YearlyProjection]()
    var balance = input.getCurrentSavings.doubleValue()
    var totalContributions = input.getCurrentSavings.doubleValue()

    for year <- 1 to yearsToRetirement do
      val startBalance = balance
      val yearlyContribution = input.getMonthlyContribution * 12
      var interestEarned = 0.0

      for _ <- 0 until 12 do
        balance += input.getMonthlyContribution
        interestEarned += balance * monthlyRate
        balance += balance * monthlyRate

      totalContributions += yearlyContribution
      projections.add(YearlyProjection(
        year,
        input.getCurrentAge + year,
        Math.round(startBalance * 100.0) / 100.0,
        Math.round(yearlyContribution * 100.0) / 100.0,
        Math.round(interestEarned * 100.0) / 100.0,
        Math.round(balance * 100.0) / 100.0
      ))

    val totalSavingsAtRetirement = balance
    val inflationFactor = Math.pow(1 + inflationRate, yearsToRetirement)
    val inflationAdjustedMonthlyIncome = input.getDesiredMonthlyIncome * inflationFactor
    val totalNeeded = inflationAdjustedMonthlyIncome * 12 * yearsInRetirement

    val withdrawalRate = if annualRate > 0 then annualRate * 0.6 else 0.04
    val monthlyIncomeFromSavings = (totalSavingsAtRetirement * withdrawalRate) / 12.0

    val savingsGap = totalNeeded - totalSavingsAtRetirement

    val requiredMonthlySavings = calculateRequiredMonthlySavings(
      totalNeeded, input.getCurrentSavings, monthlyRate, yearsToRetirement * 12)

    val riskAssessment = assessRisk(input, totalSavingsAtRetirement, totalNeeded)
    val recommendations = generateRecommendations(input, totalSavingsAtRetirement, totalNeeded, savingsGap)

    val result = RetirementResult()
    result.setTotalSavingsAtRetirement(Math.round(totalSavingsAtRetirement * 100.0) / 100.0)
    result.setTotalNeededForRetirement(Math.round(totalNeeded * 100.0) / 100.0)
    result.setMonthlyIncomeFromSavings(Math.round(monthlyIncomeFromSavings * 100.0) / 100.0)
    result.setSavingsGap(Math.round(savingsGap * 100.0) / 100.0)
    result.setOnTrack(totalSavingsAtRetirement >= totalNeeded)
    result.setYearsToRetirement(yearsToRetirement)
    result.setYearsInRetirement(yearsInRetirement)
    result.setTotalContributions(Math.round(totalContributions * 100.0) / 100.0)
    result.setTotalInterestEarned(Math.round((totalSavingsAtRetirement - totalContributions) * 100.0) / 100.0)
    result.setInflationAdjustedMonthlyIncome(Math.round(inflationAdjustedMonthlyIncome * 100.0) / 100.0)
    result.setRequiredMonthlySavings(Math.round(requiredMonthlySavings * 100.0) / 100.0)
    result.setRiskAssessment(riskAssessment)
    result.setYearlyProjections(projections)
    result.setRecommendations(recommendations)
    result

  private def calculateRequiredMonthlySavings(targetAmount: Double, currentSavings: java.lang.Double, monthlyRate: Double, months: Int): Double =
    if monthlyRate == 0 then
      return (targetAmount - currentSavings) / months
    val futureValueOfCurrent = currentSavings * Math.pow(1 + monthlyRate, months)
    val remaining = targetAmount - futureValueOfCurrent
    if remaining <= 0 then return 0
    remaining * monthlyRate / (Math.pow(1 + monthlyRate, months) - 1)

  private def assessRisk(input: RetirementInput, projected: Double, needed: Double): String =
    val ratio = projected / needed
    if ratio >= 1.5 then "LOW"
    else if ratio >= 1.0 then "MODERATE"
    else if ratio >= 0.7 then "HIGH"
    else "CRITICAL"

  private def generateRecommendations(input: RetirementInput, projected: Double, needed: Double, gap: Double): java.util.List[String] =
    val recs = new java.util.ArrayList[String]()
    val ratio = projected / needed

    if ratio < 1.0 then
      val additionalMonthly = gap / ((input.getRetirementAge - input.getCurrentAge) * 12)
      recs.add(f"Increase monthly contributions by $$$additionalMonthly%.2f to close the gap")

    if input.getExpectedAnnualReturn < 6 then
      recs.add("Consider diversifying into higher-return investments like index funds")

    if input.getRetirementAge < 65 then
      recs.add("Delaying retirement by a few years can significantly improve your outlook")

    if input.getMonthlyContribution < input.getDesiredMonthlyIncome * 0.15 then
      recs.add("Your savings rate is low relative to your income goal - aim to save at least 15% of desired income")

    if ratio >= 1.0 && ratio < 1.3 then
      recs.add("You are on track but with a thin margin - consider building a larger buffer")

    if ratio >= 1.5 then
      recs.add("You are well ahead of your retirement goal - consider early retirement or lifestyle upgrades")

    if input.getInflationRate > 4 then
      recs.add("High inflation assumption - consider inflation-protected securities (TIPS)")

    if input.getLifeExpectancy - input.getRetirementAge > 30 then
      recs.add("Long retirement horizon - ensure portfolio has growth allocation for longevity risk")

    if recs.isEmpty then
      recs.add("Your retirement plan looks solid - maintain current contributions and review annually")

    recs
