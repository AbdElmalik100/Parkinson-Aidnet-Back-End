from django_filters.filterset import FilterSet
from .models import CustomUser

class UserFilter(FilterSet):
    class Meta:
        model = CustomUser
        fields = ['username']