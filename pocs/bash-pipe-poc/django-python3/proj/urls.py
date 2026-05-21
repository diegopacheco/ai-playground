from django.urls import path
from healthapp.views import health, root

urlpatterns = [
    path("health/", health),
    path("", root),
]
