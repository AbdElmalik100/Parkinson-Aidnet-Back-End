"""
WSGI config for parkinsonsaidnet project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os
import sys
from django.core.wsgi import get_wsgi_application

sys.path.append('C:/Users/Abd El Malik/Desktop/Parkinson/Parkinson Backend/parkinsonsaidnet')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'parkinsonsaidnet.settings')

application = get_wsgi_application()


app = application
