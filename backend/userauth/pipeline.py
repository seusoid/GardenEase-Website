# backend/userauth/pipeline.py
import uuid
import logging
from social_core.exceptions import AuthException
from .models import User
from django.conf import settings
from django.shortcuts import redirect


logger = logging.getLogger(__name__)

def create_mongoengine_user(strategy, details, backend, user=None, *args, **kwargs):
    """
    social_django pipeline step to create or fetch a MongoEngine User.
    """
    # 1) If Django user already exists, just return it.
    if user:
        return {'user': user}

    email = details.get('email')
    uid   = kwargs.get('uid')
    if not email or not uid:
        raise AuthException(backend, 'Missing email or uid from Google.')

    # 2) See if we already have a MongoEngine user with this email
    existing = User.objects(email=email).first()
    if existing:
        # if they signed up by email/password before, attach the Google ID
        if not existing.google_id:
            existing.google_id = uid
            existing.auth_method = 'both'  # They now have both methods
            existing.save()
        strategy.session_set('oauth_user_email', email)
        return {'user': existing, 'is_new': False}

    # 3) Create a brand-new user, with a throw-away password that passes validation
    random_pw = uuid.uuid4().hex[:12]  # always ≥ 8 chars
    user_data = {
        'name':      details.get('fullname') or email.split('@')[0],
        'email':     email,
        'password':  random_pw,
        'google_id': uid,
        'is_admin':  email in getattr(settings, 'ADMIN_EMAILS', []),
        'auth_method': 'google',  # Mark as Google signup
        'has_set_password': False,  # They haven't set a real password yet
    }

    try:
        new_user = User(**user_data)
        new_user.save()
        logger.info(f"Created new MongoEngine user for {email}")
        strategy.session_set('oauth_user_email', email)
        return {'user': new_user, 'is_new': True}

    except Exception as e:
        logger.error(f"Error in create_mongoengine_user: {e!r}")
        raise AuthException(backend, 'Could not create user in database.')
    

def redirect_to_frontend(strategy, *args, **kwargs):
    """
    Final pipeline step: sends the user back to your front-end app
    so that oauth_success.html can postMessage() back and close.
    """
    return redirect(settings.SOCIAL_AUTH_LOGIN_REDIRECT_URL)