from django.contrib import admin
from .models import SegmentationResult


@admin.register(SegmentationResult)
class SegmentationResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'area_m2', 'created_at']
    list_filter = ['created_at']
    readonly_fields = ['created_at']

