from django.urls import path,include
from .views import signup,login_user
from . import views
from django.contrib import admin

urlpatterns = [
    path('signup/', signup, name='signup'),
     path('login/', login_user, name='login'),
     path('oauth/success/', views.oauth_success, name='oauth_success'),  # ADD THIS
    path('oauth/error/', views.oauth_error, name='oauth_error'),
    
]
