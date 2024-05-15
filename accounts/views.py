from django.shortcuts import render
from djoser.views import UserViewSet
from .models import CustomUser
from .filters import UserFilter
from google.oauth2 import id_token
from google.auth.transport import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from djoser.serializers import UserCreateSerializer
from rest_framework.authtoken.models import Token
# Create your views here.




class GoogleView(APIView):
    def post(self, request):
        try:
            # Specify the CLIENT_ID of the app that accesses the backend:
            userInfo = id_token.verify_oauth2_token(request.data.get('credential'), requests.Request(), request.data.get('client_id'))

            if userInfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Wrong issuer.')
            
            userExists = CustomUser.objects.filter(email=userInfo['email'])

            if userExists.exists():
                token, created = Token.objects.get_or_create(user=userExists[0])
                return Response({"user": userInfo, "token": token.key})
            else:
                userData = {
                    "first_name": userInfo['given_name'],
                    "last_name": userInfo['family_name'],
                    "username": userInfo['email'].split("@")[0],
                    "email": userInfo['email'],
                    "password": userInfo['email'].split("@")[0] + userInfo['sub'],
                    "re_password": userInfo['email'].split("@")[0] + userInfo['sub']
                }

                userSerializer = UserCreateSerializer(data=userData)
                userSerializer.is_valid(raise_exception=True)
                user = userSerializer.save()
                user.avatar = userInfo['picture']
                user.save()

                token, created = Token.objects.get_or_create(user=user)
                return Response({"user": userInfo, "token": token.key})
            
        except ValueError as err:
            print(err)
            content = {'message': 'Invalid token'}
            return Response(content)


class CustomUserViewSet(UserViewSet):
    filterset_class = UserFilter