import cv2
import numpy as np
import torch
from PIL import Image
from .model_loader import get_model, get_device


def predict_full_image(model, image_array, patch_size=256):
    device = get_device()
    
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        h, w, _ = image_array.shape
    else:
        raise ValueError("Изображение должно быть RGB")
    
    if image_array.max() > 1.0:
        image_array = image_array.astype(np.float32) / 255.0
    
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        padded_img = cv2.copyMakeBorder(
            (image_array * 255).astype(np.uint8), 
            0, pad_h, 0, pad_w, 
            cv2.BORDER_CONSTANT, 
            value=0
        )
        padded_img = padded_img.astype(np.float32) / 255.0
    else:
        padded_img = image_array
    
    full_pred = np.zeros((padded_img.shape[0], padded_img.shape[1]), dtype=np.float32)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    model.eval()
    with torch.no_grad():
        for y in range(0, padded_img.shape[0], patch_size):
            for x in range(0, padded_img.shape[1], patch_size):
                patch = padded_img[y:y + patch_size, x:x + patch_size]
                
                input_patch = (patch - mean) / std
                input_patch = torch.from_numpy(input_patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
                
                output = model(input_patch)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                full_pred[y:y + patch_size, x:x + patch_size] = pred
    
    final_pred_mask = full_pred[:h, :w]
    binary_pred = (final_pred_mask > 0.5).astype(np.uint8) * 255
    
    pixel_count = np.sum(binary_pred > 0)
    area_m2 = pixel_count * 0.09
    
    return binary_pred, area_m2, final_pred_mask


def process_uploaded_image(image_file):
    model = get_model()
    
    try:
        image = Image.open(image_file)
    except Exception as e:
        raise ValueError(f"Не удалось открыть изображение. Поддерживаемые форматы: PNG, JPEG, TIFF, BMP. Ошибка: {str(e)}")
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    binary_mask, area_m2, _ = predict_full_image(model, image_array)
    result_mask = Image.fromarray(binary_mask, mode='L')
    
    return result_mask, area_m2, image

