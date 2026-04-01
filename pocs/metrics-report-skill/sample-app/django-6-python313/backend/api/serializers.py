from rest_framework import serializers


class RetirementInputSerializer(serializers.Serializer):
    currentAge = serializers.IntegerField(min_value=18, max_value=80)
    retirementAge = serializers.IntegerField(min_value=50, max_value=75)
    currentSavings = serializers.FloatField(min_value=0)
    monthlyContribution = serializers.FloatField(min_value=0, max_value=100000)
    expectedAnnualReturn = serializers.FloatField(min_value=0, max_value=30)
    desiredMonthlyIncome = serializers.FloatField(min_value=500)
    lifeExpectancy = serializers.IntegerField(min_value=70, max_value=110)
    inflationRate = serializers.FloatField(min_value=0, max_value=15)
