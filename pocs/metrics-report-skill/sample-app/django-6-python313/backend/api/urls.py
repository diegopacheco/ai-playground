from django.urls import path
from api.views import CalculateView, HealthView

urlpatterns = [
    path('calculate', CalculateView.as_view()),
    path('health', HealthView.as_view()),
]
