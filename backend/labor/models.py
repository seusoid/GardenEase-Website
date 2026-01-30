from django.db import models

# Create your models here.
from mongoengine import Document, StringField, DateField, DateTimeField,IntField
from datetime import datetime

class Booking(Document):
    laborer_name = StringField(required=True)
    user_email = StringField(required=True)
    booking_date = DateField(required=True)
    time_slot = StringField(required=True, choices=["morning", "midday", "evening"])  # Changed from "early-morning" to "morning"
    service_type = StringField(required=True, choices=[
        "maintaining-garden", 
        "lawn-mowing", 
        "tree-pruning", 
        "plant-watering", 
        "weed-removal"
    ])  # Updated to match frontend options
    message = StringField()
    created_at = DateTimeField(default=datetime.utcnow)


class Review(Document):
    user_name = StringField(required=True)
    rating = IntField(min_value=1, max_value=5, required=True)
    comment = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)