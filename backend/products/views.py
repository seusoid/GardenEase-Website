import json
from django.shortcuts import render
import traceback
# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Product, CartItem


@csrf_exempt
def get_products_by_category(request, category):
    products = Product.objects(category=category)
    data = [{
        "_id": str(p.id), 
        "name": p.name,
        "price": p.price,
        "image": p.image_url
    } for p in products]
    return JsonResponse(data, safe=False)


@csrf_exempt
def add_to_cart(request):
    print("🚀 ADD TO CART VIEW CALLED")

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_email = data.get('user_email')
            product_id = data.get('product_id')
            quantity = data.get('quantity', 1)

            print("🟡 Add to Cart Request:")
            print(f"user_email: {user_email}, product_id: {product_id}, quantity: {quantity}")

            existing = CartItem.objects(user_email=user_email, product_id=product_id).first()
            if existing:
                existing.quantity += quantity
                existing.save()
                print("🟢 Updated existing item")
            else:
                new_item = CartItem(user_email=user_email, product_id=product_id, quantity=quantity)
                new_item.save()
                print("🟢 New cart item saved")

            return JsonResponse({'message': 'Product added to cart!'}, status=201)
        
        except Exception as e:
            print("❌ ERROR saving cart item:")
            traceback.print_exc()
            return JsonResponse({'error': 'Failed to add to cart'}, status=500)

    return JsonResponse({'error': 'Invalid method'}, status=400)


@csrf_exempt
def get_cart(request, email):
    if request.method == 'GET':
        cart_items = CartItem.objects(user_email=email)
        data = []

        for item in cart_items:
            # Fetch product details
            product = Product.objects(id=item.product_id).first()
            if product:
                data.append({
                    "id": str(item.id),  # cart item id ✅
                    "product_id": item.product_id,
                    "name": product.name,
                    "price": product.price,
                    "quantity": item.quantity,
                    "image": product.image_url
                })

        return JsonResponse(data, safe=False)

    return JsonResponse({'error': 'Invalid method'}, status=400)

@csrf_exempt
def delete_cart_item(request, item_id):
    if request.method == 'DELETE':
        try:
            item = CartItem.objects(id=item_id).first()
            if item:
                item.delete()
                return JsonResponse({'message': 'Deleted successfully'})
            return JsonResponse({'error': 'Item not found'}, status=404)
        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'error': 'Server error'}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=400)


@csrf_exempt
def update_cart_quantity(request, item_id):
    if request.method == 'PATCH':
        try:
            data = json.loads(request.body)
            new_quantity = data.get('quantity')

            item = CartItem.objects(id=item_id).first()
            if item and new_quantity >= 1:
                item.quantity = new_quantity
                item.save()
                return JsonResponse({'message': 'Quantity updated'}, status=200)
            else:
                return JsonResponse({'error': 'Invalid quantity or item not found'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid method'}, status=405)
