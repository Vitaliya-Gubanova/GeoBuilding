"""
ASGI config for geobuilding_segmentation project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'geobuilding_segmentation.settings')

application = get_asgi_application()

