from django.db import models

# Create your models here.
from mongoengine import Document, StringField, EmailField, BooleanField

class User(Document):
    name = StringField(required=True, max_length=100)
    email = EmailField(required=True, unique=True)
    password = StringField(required=True, min_length=8)
    is_admin = BooleanField(default=False)
    google_id  = StringField(unique=True, sparse=True)
    auth_method = StringField(choices=['email', 'google', 'both'], default='email')  # Track how they signed up
    has_set_password = BooleanField(default=False)  # Track if Google users have set a password 