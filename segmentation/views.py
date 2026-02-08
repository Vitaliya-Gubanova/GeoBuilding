from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings
import os
import io
import numpy as np
from PIL import Image
from .image_processor import process_uploaded_image
from .model_loader import load_model


def index(request):
    return render(request, 'segmentation/index.html')


def find_ground_truth_image(uploaded_filename):
    try:
        base_name = os.path.splitext(os.path.basename(uploaded_filename))[0]
        gt_dir = settings.BASE_DIR / 'Chicago_orig' / 'train' / 'gt'
        possible_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        gt_path = None
        
        for ext in possible_extensions:
            test_path = gt_dir / f"{base_name}{ext}"
            if test_path.exists():
                gt_path = test_path
                break
        
        if gt_path is None:
            for file in gt_dir.iterdir():
                if file.is_file() and base_name.lower() in file.stem.lower():
                    gt_path = file
                    break
        
        if gt_path is None or not gt_path.exists():
            return None, None
        
        gt_image = Image.open(gt_path)
        
        if gt_image.mode != 'L':
            gt_image = gt_image.convert('L')
        
        gt_array = np.array(gt_image)
        pixel_count = np.sum(gt_array > 0)
        gt_area_m2 = pixel_count * 0.09
        
        gt_image_rgb = gt_image.convert('RGB')
        
        return gt_image_rgb, gt_area_m2
        
    except Exception:
        return None, None


@require_http_methods(["POST"])
def process_image(request):
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'Изображение не загружено'}, status=400)
    
    try:
        try:
            load_model()
        except Exception as e:
            return JsonResponse({
                'error': f'Ошибка загрузки модели: {str(e)}. Убедитесь, что файл чекпоинта существует.'
            }, status=500)
        
        image_file = request.FILES['image']
        
        MAX_FILE_SIZE = 100 * 1024 * 1024
        if image_file.size > MAX_FILE_SIZE:
            return JsonResponse({
                'error': f'Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE // (1024*1024)} МБ'
            }, status=400)
        
        file_name = image_file.name.lower()
        supported_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        if not any(file_name.endswith(ext) for ext in supported_extensions):
            return JsonResponse({
                'error': f'Неподдерживаемый формат файла. Поддерживаемые форматы: {", ".join(supported_extensions)}'
            }, status=400)
        
        result_mask, area_m2, original_image = process_uploaded_image(image_file)
        gt_image, gt_area_m2 = find_ground_truth_image(image_file.name)
        
        import base64
        
        def image_to_base64(pil_image):
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response_data = {
            'success': True,
            'area_m2': round(area_m2, 2),
            'original_image': image_to_base64(original_image),
            'mask_image': image_to_base64(result_mask),
        }
        
        if gt_image is not None and gt_area_m2 is not None:
            response_data['gt_image'] = image_to_base64(gt_image)
            response_data['gt_area_m2'] = round(gt_area_m2, 2)
        
        return JsonResponse(response_data)
        
    except Exception as e:
        import traceback
        return JsonResponse({
            'error': f'Ошибка обработки: {str(e)}',
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)



