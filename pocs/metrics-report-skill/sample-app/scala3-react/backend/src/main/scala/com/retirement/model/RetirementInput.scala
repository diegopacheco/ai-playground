package com.retirement.model

import jakarta.validation.constraints.*
import scala.beans.BeanProperty

class RetirementInput:
  @BeanProperty
  @NotNull(message = "Current age is required")
  @Min(value = 18, message = "Age must be at least 18")
  @Max(value = 80, message = "Age must be at most 80")
  var currentAge: Integer = _

  @BeanProperty
  @NotNull(message = "Retirement age is required")
  @Min(value = 50, message = "Retirement age must be at least 50")
  @Max(value = 75, message = "Retirement age must be at most 75")
  var retirementAge: Integer = _

  @BeanProperty
  @NotNull(message = "Current savings is required")
  @Min(value = 0, message = "Current savings cannot be negative")
  var currentSavings: java.lang.Double = _

  @BeanProperty
  @NotNull(message = "Monthly contribution is required")
  @Min(value = 0, message = "Monthly contribution cannot be negative")
  @Max(value = 100000, message = "Monthly contribution cannot exceed 100,000")
  var monthlyContribution: java.lang.Double = _

  @BeanProperty
  @NotNull(message = "Expected annual return is required")
  @Min(value = 0, message = "Expected return cannot be negative")
  @Max(value = 30, message = "Expected return cannot exceed 30%")
  var expectedAnnualReturn: java.lang.Double = _

  @BeanProperty
  @NotNull(message = "Desired monthly income is required")
  @Min(value = 500, message = "Desired monthly income must be at least 500")
  var desiredMonthlyIncome: java.lang.Double = _

  @BeanProperty
  @NotNull(message = "Life expectancy is required")
  @Min(value = 70, message = "Life expectancy must be at least 70")
  @Max(value = 110, message = "Life expectancy must be at most 110")
  var lifeExpectancy: Integer = _

  @BeanProperty
  @NotNull(message = "Inflation rate is required")
  @Min(value = 0, message = "Inflation rate cannot be negative")
  @Max(value = 15, message = "Inflation rate cannot exceed 15%")
  var inflationRate: java.lang.Double = _
