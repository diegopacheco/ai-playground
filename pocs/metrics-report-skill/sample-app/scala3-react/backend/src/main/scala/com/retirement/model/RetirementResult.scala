package com.retirement.model

import scala.beans.BeanProperty

class RetirementResult:
  @BeanProperty var totalSavingsAtRetirement: Double = _
  @BeanProperty var totalNeededForRetirement: Double = _
  @BeanProperty var monthlyIncomeFromSavings: Double = _
  @BeanProperty var savingsGap: Double = _
  @BeanProperty var onTrack: Boolean = _
  @BeanProperty var yearsToRetirement: Int = _
  @BeanProperty var yearsInRetirement: Int = _
  @BeanProperty var totalContributions: Double = _
  @BeanProperty var totalInterestEarned: Double = _
  @BeanProperty var inflationAdjustedMonthlyIncome: Double = _
  @BeanProperty var requiredMonthlySavings: Double = _
  @BeanProperty var riskAssessment: String = _
  @BeanProperty var yearlyProjections: java.util.List[YearlyProjection] = _
  @BeanProperty var recommendations: java.util.List[String] = _
