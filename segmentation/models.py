from django.db import models
from django.utils import timezone


class SegmentationResult(models.Model):
    original_image = models.ImageField(upload_to='uploads/')
    result_mask = models.ImageField(upload_to='results/')
    area_m2 = models.FloatField(help_text="Площадь застройки в квадратных метрах")
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Результат сегментации'
        verbose_name_plural = 'Результаты сегментации'
    
    def __str__(self):
        return f"Сегментация от {self.created_at.strftime('%Y-%m-%d %H:%M')} - {self.area_m2:.2f} м²"
