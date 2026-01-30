from django.shortcuts import render

# Create your views here.
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.contrib.auth import authenticate, login
from django.contrib.auth import get_user_model
from django.views.decorators.csrf import csrf_exempt
from .models import User
from django.conf import settings
from django.contrib.auth.hashers import make_password
from django.contrib.auth.hashers import check_password
from django.shortcuts import render



@csrf_exempt
def signup(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        if User.objects(email=email).first():
            return JsonResponse({'error': 'Email already exists'}, status=400)

        hashed_pw = make_password(password)
        user = User(name=name, email=email, password=hashed_pw)
        # ← new line: flag admin accounts
        user.is_admin = (email in settings.ADMIN_EMAILS)
        user.save()
        print(f"[Signup] email={email!r}, is_admin={user.is_admin}")

        return JsonResponse({
            'message': 'Signup successful!',
            'is_admin': user.is_admin   # optional
        }, status=201)
    

    return JsonResponse({'error': 'Invalid request method'}, status=400)




@csrf_exempt
def set_password(request):
    """Allow Google users to set a password for email/password login"""
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        new_password = data.get('password')
        
        if not email or not new_password:
            return JsonResponse({'error': 'Email and password required'}, status=400)
            
        if len(new_password) < 8:
            return JsonResponse({'error': 'Password must be at least 8 characters'}, status=400)

        user = User.objects(email=email).first()
        if not user:
            return JsonResponse({'error': 'User not found'}, status=404)
            
        if not user.google_id:
            return JsonResponse({'error': 'This feature is only for Google users'}, status=400)

        # Update password and auth method
        user.password = make_password(new_password)
        user.has_set_password = True
        user.auth_method = 'both'
        user.save()

        return JsonResponse({'message': 'Password set successfully'}, status=200)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email = data.get('email')
        password = data.get('password')

        user = User.objects(email=email).first()
        if not user:
            return JsonResponse({'error': 'Email not registered'}, status=404)
        
        # Check if user signed up via Google but hasn't set a password
        if user.google_id and not user.has_set_password:
            return JsonResponse({
                'error': 'This account was created with Google. Please use "Sign in with Google" or set a password first.',
                'needs_password_setup': True
            }, status=400)
        
        if check_password(password, user.password):
            return JsonResponse({
                'message': 'Login successful!',
                'name': user.name,
                'email': user.email,
                'is_admin': user.is_admin 
            }, status=200)
        else:
            return JsonResponse({'error': 'Invalid password'}, status=401)

    return JsonResponse({'error': 'Invalid request method'}, status=400)





def oauth_success(request):
    # retrieve the Google‐signed up user's email from the session
    email = request.session.pop('oauth_user_email', None)
    user_data = {}
    if email:
        try:
            u = User.objects(email=email).first()
            user_data = {
                'id': str(u.id),
                'name': u.name,
                'email': u.email,
                'is_admin': u.is_admin,  # Make sure this is included
            }
        except Exception:
            user_data = {}

    return render(request, 'oauth_success.html', {
        'user_data': json.dumps(user_data)
    })

def oauth_error(request):
    error_msg = request.GET.get('message', 'Unknown error')
    return render(request, 'oauth_error.html', {
        'error': error_msg
    })
# import logging
# logger = logging.getLogger(__name__)

# def oauth_success(request):
#     logger.info(f"OAuth success called. User: {request.user}")
#     logger.info(f"Session: {request.session.items()}")
#     # ... rest of your code

