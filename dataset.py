# Данный код готовит данные для обучения.
# Обучать и верифицировать модель будем только для города Чикаго.
# Берется базовый сет снимков и из него выбираются изображения только для города Чикаго.
# Оригинальные снимки и оригинальные маски 5000х5000 разбиваются на картинки меньшего размера 256х256 - патчи.
# Излишки от оригинального снимка при этом игнорируются.
# Созданный таким образом датасет из патчей архивируется (chicago_patches.7z) и по этим кусочкам-патчам происходит
# обучение модели.
import cv2
import os
from tqdm import tqdm
import numpy as np

# Пути к исходному датасету (изменить, если папки лежат в другом месте)
base_path = r'D:\Magellan\Pyton\MFTI\Sputnik\AerialImageDataset\train'
images_dir = os.path.join(base_path, 'images')
masks_dir = os.path.join(base_path, 'gt')

# Пути куда сохранить патчи
output_images_dir = 'chicago_patches/images'
output_masks_dir = 'chicago_patches/masks'

# Создаем папки, если их нет
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

PATCH_SIZE = 256
CITY_NAME = 'chicago'

# Получаем список файлов только для Чикаго
image_files = [f for f in os.listdir(images_dir) if f.startswith(CITY_NAME) and f.endswith('.tif')]

print(f"Найдено {len(image_files)} больших снимков для города {CITY_NAME}")

patch_count = 0
skipped_empty = 0

for filename in tqdm(image_files):
    # Загружаем большое изображение и маску
    img = cv2.imread(os.path.join(images_dir, filename))
    mask = cv2.imread(os.path.join(masks_dir, filename), cv2.IMREAD_GRAYSCALE)

    # 5000 // 256 = 19 полных патчей
    n_patches = 5000 // PATCH_SIZE

    for i in range(n_patches):
        for j in range(n_patches):
            y = i * PATCH_SIZE
            x = j * PATCH_SIZE

            img_patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            # Проверка на наличие зданий
            # Считаем количество белых пикселей (зданий)
            building_pixels = np.sum(mask_patch > 0)

            # Логика фильтрации:
            # Если зданий нет (пустой патч), сохраняем его только в 10% случаев
            # Это поможет сбалансировать датасет
            if building_pixels == 0:
                if np.random.random() > 0.1:  # 90% пустых патчей пропускаем
                    skipped_empty += 1
                    continue

            # Формируем имя файла
            patch_name = f"{filename.replace('.tif', '')}_patch_{i}_{j}.png"

            # Сохраняем
            cv2.imwrite(os.path.join(output_images_dir, patch_name), img_patch)
            cv2.imwrite(os.path.join(output_masks_dir, patch_name), mask_patch)
            patch_count += 1

print(f"\nГотово!")
print(f"Сохранено патчей: {patch_count}")
print(f"Пропущено пустых патчей для баланса: {skipped_empty}")
print(f"Итоговые данные лежат в папке: chicago_patches")
