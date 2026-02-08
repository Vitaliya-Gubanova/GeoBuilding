import torch
import torch.nn as nn
from pathlib import Path
from django.conf import settings
import os


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # Энкодер
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        # Центр (Bottleneck)
        self.bottleneck = DoubleConv(512, 1024)

        # Декодер
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        sk1, x = self.down1(x)
        sk2, x = self.down2(x)
        sk3, x = self.down3(x)
        sk4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, sk4)
        x = self.up2(x, sk3)
        x = self.up3(x, sk2)
        x = self.up4(x, sk1)

        return self.out(x)


_model = None
_device = None


def get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def load_model(checkpoint_path=None):
    global _model
    
    if _model is not None:
        return _model
    
    if checkpoint_path is None:
        checkpoint_path = settings.MODEL_CHECKPOINT_PATH
    
    if not os.path.exists(checkpoint_path):
        base_dir = Path(checkpoint_path).parent
        checkpoints = list(base_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            try:
                print(f"Используется чекпоинт: {str(checkpoint_path)}")
            except UnicodeEncodeError:
                print(f"Используется чекпоинт: {checkpoint_path.name}")
        else:
            error_msg = f"Чекпоинт модели не найден: {str(checkpoint_path)}"
            raise FileNotFoundError(error_msg)
    
    device = get_device()
    model = UNet().to(device)
    
    try:
        checkpoint_path_str = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path_str, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        _model = model
        try:
            print(f"Модель успешно загружена из {checkpoint_path_str}")
        except UnicodeEncodeError:
            print("Модель успешно загружена")
        return model
    except Exception as e:
        error_msg = str(e)
        if "'charmap' codec" in error_msg or "encode" in error_msg.lower():
            raise Exception("Ошибка при загрузке модели: проблема с кодировкой файла или пути. Убедитесь, что путь к чекпоинту не содержит специальных символов.")
        raise Exception(f"Ошибка при загрузке модели: {error_msg}")


def get_model():
    if _model is None:
        return load_model()
    return _model

