from django.test import TestCase
from api.service import RetirementCalculationService


def valid_input():
    return {
        'currentAge': 30,
        'retirementAge': 65,
        'currentSavings': 50000,
        'monthlyContribution': 1000,
        'expectedAnnualReturn': 7,
        'desiredMonthlyIncome': 5000,
        'lifeExpectancy': 90,
        'inflationRate': 3,
    }


class RetirementCalculationServiceTest(TestCase):

    def test_calculate_returns_result(self):
        result = RetirementCalculationService.calculate(valid_input())
        self.assertIsNotNone(result)

    def test_years_to_retirement(self):
        result = RetirementCalculationService.calculate(valid_input())
        self.assertEqual(result['yearsToRetirement'], 35)

    def test_years_in_retirement(self):
        result = RetirementCalculationService.calculate(valid_input())
        self.assertEqual(result['yearsInRetirement'], 25)

    def test_retirement_age_must_be_greater_than_current(self):
        data = valid_input()
        data['currentAge'] = 70
        with self.assertRaises(ValueError):
            RetirementCalculationService.calculate(data)

    def test_at_least_5_years_gap(self):
        data = valid_input()
        data['currentAge'] = 62
        with self.assertRaises(ValueError):
            RetirementCalculationService.calculate(data)

    def test_life_expectancy_greater_than_retirement(self):
        data = valid_input()
        data['lifeExpectancy'] = 60
        with self.assertRaises(ValueError):
            RetirementCalculationService.calculate(data)

    def test_return_greater_than_inflation(self):
        data = valid_input()
        data['inflationRate'] = 10
        with self.assertRaises(ValueError):
            RetirementCalculationService.calculate(data)

    def test_projection_count_matches_years(self):
        result = RetirementCalculationService.calculate(valid_input())
        self.assertEqual(len(result['yearlyProjections']), 35)

    def test_total_contributions_include_initial(self):
        result = RetirementCalculationService.calculate(valid_input())
        self.assertGreater(result['totalContributions'], 50000)

    def test_recommendations_not_empty(self):
        result = RetirementCalculationService.calculate(valid_input())
        self.assertTrue(len(result['recommendations']) > 0)
