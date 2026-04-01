package com.retirement.service

import com.retirement.model.RetirementInput
import org.junit.jupiter.api.{BeforeEach, Test}
import org.junit.jupiter.api.Assertions.*

class RetirementCalculationServiceTest:

  var service: RetirementCalculationService = _

  @BeforeEach
  def setUp(): Unit =
    service = RetirementCalculationService()

  private def createValidInput(): RetirementInput =
    val input = RetirementInput()
    input.currentAge = 30
    input.retirementAge = 65
    input.currentSavings = 50000.0
    input.monthlyContribution = 1000.0
    input.expectedAnnualReturn = 7.0
    input.desiredMonthlyIncome = 5000.0
    input.lifeExpectancy = 90
    input.inflationRate = 3.0
    input

  @Test
  def testValidCalculationReturnsResult(): Unit =
    val result = service.calculate(createValidInput())
    assertNotNull(result)
    assertTrue(result.totalSavingsAtRetirement > 0)

  @Test
  def testYearsToRetirementCalculation(): Unit =
    val result = service.calculate(createValidInput())
    assertEquals(35, result.yearsToRetirement)

  @Test
  def testYearsInRetirementCalculation(): Unit =
    val result = service.calculate(createValidInput())
    assertEquals(25, result.yearsInRetirement)

  @Test
  def testRetirementAgeNotGreaterThanCurrentAge(): Unit =
    val input = createValidInput()
    input.retirementAge = 25
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testLessThanFiveYearsToRetirement(): Unit =
    val input = createValidInput()
    input.currentAge = 62
    input.retirementAge = 65
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testLifeExpectancyLessThanRetirementAge(): Unit =
    val input = createValidInput()
    input.lifeExpectancy = 60
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testReturnLessThanInflation(): Unit =
    val input = createValidInput()
    input.expectedAnnualReturn = 2.0
    input.inflationRate = 5.0
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testProjectionsCountMatchesYears(): Unit =
    val result = service.calculate(createValidInput())
    assertEquals(35, result.yearlyProjections.size())

  @Test
  def testTotalContributionsIncludeInitialSavings(): Unit =
    val result = service.calculate(createValidInput())
    val expectedContributions = 50000.0 + (1000.0 * 12 * 35)
    assertEquals(expectedContributions, result.totalContributions, 0.01)

  @Test
  def testRecommendationsNotEmpty(): Unit =
    val result = service.calculate(createValidInput())
    assertNotNull(result.recommendations)
    assertFalse(result.recommendations.isEmpty)
