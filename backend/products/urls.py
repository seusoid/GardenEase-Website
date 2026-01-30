from django.urls import path
from . import views

from .views import get_products_by_category, add_to_cart, get_cart

urlpatterns = [
    path('add/', add_to_cart),                 
    path('cart/<str:email>/', get_cart),          
    path('<str:category>/', get_products_by_category), 
    path('cart/delete/<str:item_id>/', views.delete_cart_item),
    path('cart/update/<str:item_id>/', views.update_cart_quantity),

]

