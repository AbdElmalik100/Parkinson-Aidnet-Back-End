from rest_framework import serializers
from .models import *


class ContactEmailsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContactEmails
        fields = '__all__'

class DrawingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Drawing
        fields = '__all__'

class MRISerializer(serializers.ModelSerializer):
    class Meta:
        model = MRI
        fields = '__all__'

class VoiceBiometricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = VoiceBiometrics
        fields = '__all__'

class TappyKeyboardSerializer(serializers.ModelSerializer):
    class Meta:
        model = TappyKeyboard
        fields = '__all__'