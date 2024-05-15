from rest_framework import serializers
from django.contrib.auth import get_user_model
from djoser.serializers import UserSerializer
from .filters import UserFilter


class CustomUserSerializer(UserSerializer):
    filterset_class = UserFilter
    class Meta:
        model = get_user_model()
        exclude = ['groups', 'user_permissions', 'is_active', 'is_staff', 'is_superuser', 'password']
