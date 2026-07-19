from django.urls import path

from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("simulation/new", views.simulation_new, name="simulation_new"),
    path("simulation/advance", views.simulation_advance, name="simulation_advance"),
    path("ai/provider", views.provider_select, name="provider_select"),
    path("ai/new", views.ai_new, name="ai_new"),
    path("ai/reveal", views.ai_reveal, name="ai_reveal"),
    path("ai/move", views.ai_move, name="ai_move"),
]
