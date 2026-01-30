from django.urls import path
from . import views

urlpatterns = [
    path('', views.plant_page, name='plant_page'),
    path('detect/', views.detect_plant, name='detect_plant'),
]

