# Generated by Django 4.2.9 on 2024-05-02 23:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0008_alter_voicebiometrics_audio'),
    ]

    operations = [
        migrations.AddField(
            model_name='voicebiometrics',
            name='parkinson',
            field=models.BooleanField(blank=True, default=False, null=True),
        ),
    ]
