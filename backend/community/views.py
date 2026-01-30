from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from community.models import CommunityPost, Reply
import json

@csrf_exempt
def create_post(request):
    if request.method == 'POST':
        user_email = request.POST.get('user_email')
        category = request.POST.get('category')
        text = request.POST.get('text')
        image = request.FILES.get('image')

        image_url = None
        if image:
            fs = FileSystemStorage(location='media/community_posts/')
            filename = fs.save(image.name, image)
            image_url = f"/media/community_posts/{filename}"

        post = CommunityPost(
            user_email=user_email,
            category=category,
            text=text,
            image_url=image_url
        )
        post.save()
        return JsonResponse({'message': 'Post created successfully'}, status=201)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def create_reply(request):
    if request.method == 'POST':
        post_id = request.POST.get('post_id')
        user_email = request.POST.get('user_email')
        text = request.POST.get('text')
        image = request.FILES.get('image')

        image_url = None
        if image:
            fs = FileSystemStorage(location='media/replies/')
            filename = fs.save(image.name, image)
            image_url = f"/media/replies/{filename}"

        reply = Reply(
            post=CommunityPost.objects.get(id=post_id),
            user_email=user_email,
            text=text,
            image_url=image_url
        )
        reply.save()
        return JsonResponse({'message': 'Reply added successfully'}, status=201)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def get_posts(request):
    if request.method == 'GET':
        posts_data = []
        for post in CommunityPost.objects.order_by('-created_at'):
            replies = Reply.objects(post=post).order_by('created_at')
            posts_data.append({
                'id': str(post.id),
                'user_email': post.user_email,
                'category': post.category,
                'text': post.text,
                'image_url': request.build_absolute_uri(post.image_url),

                'created_at': post.created_at.isoformat(),
                'replies': [
                    {
                        'user_email': r.user_email,
                        'text': r.text,
                        'image_url': r.image_url,
                        'created_at': r.created_at.isoformat()
                    }
                    for r in replies
                ]
            })
        return JsonResponse({'posts': posts_data}, safe=False)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
