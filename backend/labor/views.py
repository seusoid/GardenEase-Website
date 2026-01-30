from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from labor.models import Booking,Review
import datetime
import json
from userauth.models import User


@csrf_exempt
def book_labor(request):
    if request.method == 'POST':
        try:
            laborer_name = request.POST.get('laborer_name')
            user_email = request.POST.get('user_email')
            booking_date = request.POST.get('booking_date')
            time_slot = request.POST.get('time_slot')
            service_type = request.POST.get('service_type')
            message = request.POST.get('message', '')

            # Convert booking_date to a real date
            booking_date = datetime.datetime.strptime(booking_date, '%Y-%m-%d').date()

            booking = Booking(
                laborer_name=laborer_name,
                user_email=user_email,
                booking_date=booking_date,
                time_slot=time_slot,
                service_type=service_type,
                message=message
            )
            booking.save()

            return JsonResponse({'message': 'Booking successful!'}, status=201)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


from labor.models import Review  # Make sure this import is at the top

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@csrf_exempt
def submit_review(request):
    if request.method == 'POST':
        try:
            user_name = request.POST.get('user_name')
            rating = int(request.POST.get('rating', 0))
            comment = request.POST.get('comment')

            if not (user_name and rating and comment):
                return JsonResponse({'error': 'Missing fields'}, status=400)

            if rating < 1 or rating > 5:
                return JsonResponse({'error': 'Rating must be 1 to 5'}, status=400)

            review = Review(
                user_name=user_name,
                rating=rating,
                comment=comment
            )
            review.save()

            return JsonResponse({'message': 'Review submitted successfully'}, status=201)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)



def get_reviews(request):
    if request.method == 'GET':
        try:
            reviews = Review.objects.order_by('-created_at')  # latest first
            review_list = []

            for r in reviews:
                review_list.append({
                    "user_name": r.user_name,
                    "rating": r.rating,
                    "comment": r.comment,
                    "created_at": r.created_at.strftime('%Y-%m-%d %H:%M')
                })

            return JsonResponse({'reviews': review_list}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)



@csrf_exempt
def booking_history(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
        email = data.get('email')

        # lookup user and check admin flag
        user = User.objects(email=email).first()
        if not user or not user.is_admin:
            return JsonResponse({'error': 'Not authorized'}, status=403)

        # fetch all bookings, newest first
        bookings_qs = Booking.objects.order_by('-created_at')  # Order by created_at instead of booking_date
        history = []
        for b in bookings_qs:
            history.append({
                'id': str(b.id), 
                'laborer_name':  b.laborer_name,
                'user_email':    b.user_email,
                'booking_date':  b.booking_date.strftime('%Y-%m-%d'),
                'time_slot':     b.time_slot,
                'service_type':  b.service_type,
                'message':       b.message or '',  # Ensure message is never None
                'created_at':    b.created_at.strftime('%Y-%m-%d %H:%M:%S'),  # Changed from 'booked_on' to 'created_at'
            })

        return JsonResponse({'bookings': history}, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def booking_detail(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    data = json.loads(request.body)
    email = data.get('email')
    booking_id = data.get('id')

    user = User.objects(email=email).first()
    if not user or not user.is_admin:
        return JsonResponse({'error': 'Not authorized'}, status=403)

    b = Booking.objects(id=booking_id).first()
    if not b:
        return JsonResponse({'error': 'Booking not found'}, status=404)

    return JsonResponse({
        'id':           str(b.id),
        'laborer_name': b.laborer_name,
        'user_email':   b.user_email,
        'booking_date': b.booking_date.strftime('%Y-%m-%d'),
        'time_slot':    b.time_slot,
        'service_type': b.service_type,
        'message':      b.message or '',
        'created_at':   b.created_at.strftime('%Y-%m-%d %H:%M:%S'),
    })


@csrf_exempt
def cancel_booking(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
        email = data.get('email')  # Admin's email
        user_email = data.get('user_email')  # User whose booking to cancel
        service_type = data.get('service_type')  # Service type to identify booking

        # Check if the requester is admin
        user = User.objects(email=email).first()
        if not user or not user.is_admin:
            return JsonResponse({'error': 'Not authorized'}, status=403)

        # Find and delete the booking based on user_email and service_type
        # Convert service type from display format back to database format
        service_type_db = service_type.lower().replace(' ', '-')
        
        deleted = Booking.objects(user_email=user_email, service_type=service_type_db).delete()
        
        if deleted:
            return JsonResponse({'message': f'Booking for {service_type} by {user_email} has been removed successfully'}, status=200)
        else:
            return JsonResponse({'error': 'Booking not found'}, status=404)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)