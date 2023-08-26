# Generated by Django 4.2.3 on 2023-08-26 08:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Diary',
            fields=[
                ('idx', models.AutoField(primary_key=True, serialize=False)),
                ('writer', models.CharField(max_length=20)),
                ('subject', models.CharField(max_length=200)),
                ('content', models.TextField()),
                ('create_date', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='Comment',
            fields=[
                ('idx', models.AutoField(primary_key=True, serialize=False)),
                ('commenter', models.CharField(blank=True, max_length=20, null=True)),
                ('content', models.TextField()),
                ('create_date', models.DateTimeField(auto_now=True)),
                ('diary_idx', models.ForeignKey(db_column='diary_idx', on_delete=django.db.models.deletion.CASCADE, related_name='diary', to='diary.diary')),
            ],
        ),
    ]
