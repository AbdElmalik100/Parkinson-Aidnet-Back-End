# Generated by Django 5.0.3 on 2024-05-11 17:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0015_alter_mri_heatmap_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mri',
            name='heatmap_image',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
