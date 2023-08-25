from django.db import models


class Diary(models.Model):
    idx = models.AutoField(primary_key=True)
    writer = models.CharField(max_length=20, blank=False, null=False)
    subject = models.CharField(max_length=200)
    content = models.TextField(null=False)
    create_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.subject


class Comment(models.Model):
    diary = models.ForeignKey(Diary, on_delete=models.CASCADE)
    commenter = models.CharField(max_length=20, blank=True, null=True)
    content = models.TextField(null=False)
    create_date = models.DateTimeField(auto_now=True)
