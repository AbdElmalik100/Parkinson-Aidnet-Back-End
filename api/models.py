from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

# Create your models here.

# Validators for wav file format
def validate_wav_file(value):
    if not value.name.endswith('.wav'):
        raise ValidationError(
            _('File type not supported. Only WAV files are allowed.'),
            params={'value': value},
        )


class ContactEmails(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    first_name = models.CharField(max_length = 255)
    last_name = models.CharField(max_length = 255)
    email = models.EmailField(max_length = 255)
    subject = models.CharField(max_length = 255)
    message =  models.TextField()





class Drawing(models.Model):    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    result = models.CharField(max_length=255, blank=True, null=True)
    image = models.ImageField(upload_to='drawing')
    parkinson = models.BooleanField(default=False, blank=True, null=True)

class MRI(models.Model):    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    result = models.CharField(max_length=255, blank=True, null=True)
    image = models.ImageField(upload_to='mri')
    heatmap_image = models.CharField(max_length=255, blank=True, null=True)
    parkinson = models.BooleanField(default=False, blank=True, null=True)


class VoiceBiometrics(models.Model):    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    result = models.CharField(max_length=255, blank=True, null=True)
    audio = models.FileField(upload_to='audio', validators=[validate_wav_file])
    parkinson = models.BooleanField(default=False, blank=True, null=True)

class TappyKeyboard(models.Model):    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    result = models.CharField(max_length=255, blank=True, null=True)
    file = models.FileField(upload_to='tappy')
    parkinson = models.BooleanField(default=False, blank=True, null=True)

    
