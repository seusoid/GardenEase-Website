from django.shortcuts import render
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from ultralytics import YOLO

# Load model only once
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_model', 'best.pt')


model = YOLO(MODEL_PATH)

@csrf_exempt
def detect_plant(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        file_path = default_storage.save('temp/' + img_file.name, img_file)
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            results = model.predict(full_file_path)

            # Get class name and confidence
            class_name = results[0].names[results[0].probs.top1]
            confidence = float(results[0].probs.top1conf)

            if os.path.exists(full_file_path):
                os.remove(full_file_path)

            return JsonResponse({
                'class_name': class_name,
                'confidence': confidence
            })

        except Exception as e:
            print("Error during prediction:", e)
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def plant_page(request):
    return render(request, 'plant_detection/index.html')


def detect_plant_from_path(image_path):
    """
    Detect plant from an image path using the loaded YOLO model.
    """
    if not os.path.exists(image_path):
        return {'error': 'Image file does not exist'}

    results = model.predict(image_path)

    class_name = results[0].names[results[0].probs.top1]
    confidence = float(results[0].probs.top1conf)

    return {
        'class_name': class_name,
        'confidence': confidence
    }


#image_path="X:/dataset_on_plants_with_more_than_1100_pictures/dataset_on_plants_with_more_than_1100_pictures/Acacia dealbata Link/0a93da02e079565ada9958d01799acc1d2ab37f5.jpg"
#detect_plant_from_path(image_path)