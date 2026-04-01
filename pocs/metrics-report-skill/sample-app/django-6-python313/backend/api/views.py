from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from api.serializers import RetirementInputSerializer
from api.service import RetirementCalculationService


class CalculateView(APIView):
    def post(self, request):
        serializer = RetirementInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({'errors': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = RetirementCalculationService.calculate(serializer.validated_data)
            return Response(result)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class HealthView(APIView):
    def get(self, request):
        return Response({'status': 'UP'})
