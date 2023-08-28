from django.db import models


class Diary(models.Model):
    idx = models.AutoField(primary_key=True)
    writer = models.CharField(max_length=20, blank=False, null=False)
    subject = models.CharField(max_length=200)
    content = models.TextField(null=False)
    create_date = models.DateTimeField(auto_now=True)
    update_date = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.subject


class Comment(models.Model):
    idx = models.AutoField(primary_key=True)
    diary_idx = models.ForeignKey('Diary', related_name='diary_idx', on_delete=models.CASCADE, db_column='diary_idx')
    commenter = models.CharField(max_length=20, blank=True, null=True)
    content = models.TextField(null=False)
    create_date = models.DateTimeField(auto_now=True)
