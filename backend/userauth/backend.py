# from social_core.backends.google import GoogleOAuth2
# from social_core.exceptions import AuthException
# from .models import User

# class MongoEngineGoogleOAuth2(GoogleOAuth2):
#     """Custom Google OAuth2 backend for MongoEngine User model"""
    
#     def get_user(self, user_id):
#         """Retrieve user by ID"""
#         try:
#             return User.objects(google_id=user_id).first()
#         except User.DoesNotExist:
#             return None
    
#     def create_user(self, *args, **kwargs):
#         """This method is overridden by our pipeline"""
#         return None
pass