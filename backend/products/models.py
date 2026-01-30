from django.db import models

# Create your models here.
from mongoengine import Document, StringField, FloatField, ReferenceField, IntField, DateTimeField
from datetime import datetime
class Product(Document):
    name = StringField(required=True)
    price = FloatField(required=True)
    category = StringField(required=True)  # e.g. 'fragrant', 'indoor'
    image_url = StringField(required=True)


class CartItem(Document):
    user_email = StringField(required=True)
    product_id = StringField(required=True)  # we’ll store it as str(product.id)
    quantity = IntField(default=1)
    added_at = DateTimeField(default=datetime.utcnow)