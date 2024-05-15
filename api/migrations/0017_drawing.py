# Generated by Django 5.0.3 on 2024-05-13 21:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0016_alter_mri_heatmap_image'),
    ]

    operations = [
        migrations.CreateModel(
            name='Drawing',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('result', models.CharField(blank=True, max_length=255, null=True)),
                ('image', models.ImageField(upload_to='drawing')),
                ('parkinson', models.BooleanField(blank=True, default=False, null=True)),
            ],
        ),
    ]
