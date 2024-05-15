from rest_framework.routers import DefaultRouter
from .views import *
from django.urls import path, include
# from .views import 


router = DefaultRouter()
router.register('contact', ContactEmailViewSet)
router.register('drawing', DrawingViewSet)
router.register('mri', MRIViewSet)
router.register('voice_biometrics', VoiceBiometricsViewSet)
router.register('tappy_keyboard', TappyKeyboardViewSet)

urlpatterns = [
    path('api/', include(router.urls))
]
