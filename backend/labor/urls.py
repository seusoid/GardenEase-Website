from django.urls import path
from .views import book_labor, submit_review, get_reviews, booking_history, booking_detail, cancel_booking

urlpatterns = [
    path('book/', book_labor, name='book_labor'),
    path('submit_review/', submit_review, name='submit_review'), 
    path('reviews/', get_reviews, name='get_reviews'),
    path('booking-history/', booking_history, name='booking_history'),
    path('booking-history/', booking_history, name='booking_history'),
    path('booking-detail/', booking_detail, name='booking_detail'),
    path('cancel-booking/', cancel_booking, name='cancel_booking'), 
]
