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
    input.setCurrentAge(30)
    input.setRetirementAge(65)
    input.setCurrentSavings(50000.0)
    input.setMonthlyContribution(1000.0)
    input.setExpectedAnnualReturn(7.0)
    input.setDesiredMonthlyIncome(5000.0)
    input.setLifeExpectancy(90)
    input.setInflationRate(3.0)
    input

  @Test
  def testValidCalculationReturnsResult(): Unit =
    val result = service.calculate(createValidInput())
    assertNotNull(result)
    assertTrue(result.getTotalSavingsAtRetirement > 0)

  @Test
  def testYearsToRetirementCalculation(): Unit =
    val result = service.calculate(createValidInput())
    assertEquals(35, result.getYearsToRetirement)

  @Test
  def testYearsInRetirementCalculation(): Unit =
    val result = service.calculate(createValidInput())
    assertEquals(25, result.getYearsInRetirement)

  @Test
  def testRetirementAgeNotGreaterThanCurrentAge(): Unit =
    val input = createValidInput()
    input.setRetirementAge(25)
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testLessThanFiveYearsToRetirement(): Unit =
    val input = createValidInput()
    input.setCurrentAge(62)
    input.setRetirementAge(65)
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testLifeExpectancyLessThanRetirementAge(): Unit =
    val input = createValidInput()
    input.setLifeExpectancy(60)
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testReturnLessThanInflation(): Unit =
    val input = createValidInput()
    input.setExpectedAnnualReturn(2.0)
    input.setInflationRate(5.0)
    assertThrows(classOf[IllegalArgumentException], () => service.calculate(input))

  @Test
  def testProjectionsCountMatchesYears(): Unit =
    val result = service.calculate(createValidInput())
    assertEquals(35, result.getYearlyProjections.size())

  @Test
  def testTotalContributionsIncludeInitialSavings(): Unit =
    val result = service.calculate(createValidInput())
    val expectedContributions = 50000.0 + (1000.0 * 12 * 35)
    assertEquals(expectedContributions, result.getTotalContributions, 0.01)

  @Test
  def testRecommendationsNotEmpty(): Unit =
    val result = service.calculate(createValidInput())
    assertNotNull(result.getRecommendations)
    assertFalse(result.getRecommendations.isEmpty)
