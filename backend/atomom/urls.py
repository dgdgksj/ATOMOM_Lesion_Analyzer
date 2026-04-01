from django.urls import path

from . import views

urlpatterns = [
    path("", views.analyze_lesion, name="lesion_analyzer"),
]
