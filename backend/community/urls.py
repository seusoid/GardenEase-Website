from django.urls import path
from .views import create_post, create_reply, get_posts

urlpatterns = [
    path('create_post/', create_post, name='create_post'),
    path('create_reply/', create_reply, name='create_reply'),
    path('get_posts/', get_posts, name='get_posts'),
]
