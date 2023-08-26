from django.db import models


# Create your models here.
class Users(models.Model):
  id = models.BigAutoField(primary_key=True)
  userid = models.CharField(max_length=50, null=False, verbose_name='유저 아이디')
  passwd = models.CharField(max_length=500, null=False, verbose_name='유저 비밀번호')
  name = models.CharField(max_length=20, null=False, verbose_name='유저 닉네임')
  email = models.EmailField(max_length=128, unique=True, verbose_name='유저 이메일')
  tel = models.CharField(max_length=20, null=True)
  register_dttm = models.DateTimeField(auto_now_add=True, verbose_name='계정 생성 시간')

  def __str__(self):
    return self.name
