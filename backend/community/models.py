from django.db import models

# Create your models here.
from mongoengine import Document, StringField, DateTimeField, ReferenceField, ListField
from datetime import datetime

# One post in the community section
class CommunityPost(Document):
    user_email = StringField(required=True)
    category = StringField(required=True, choices=["suggestions", "sharing", "problems","design",])
    text = StringField()
    image_url = StringField()  # Optional image
    created_at = DateTimeField(default=datetime.utcnow)

# A reply to a post
class Reply(Document):
    post = ReferenceField(CommunityPost, reverse_delete_rule=2)  # CASCADE delete
    user_email = StringField(required=True)
    text = StringField()
    image_url = StringField()  # Optional
    created_at = DateTimeField(default=datetime.utcnow)
