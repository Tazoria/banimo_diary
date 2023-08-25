from django.contrib import admin
from .models import Users


# admin 페이지에서 보기
@admin.register(Users)
class UsersAdmin(admin.ModelAdmin):
  list_display = (
    'id',
    'userid',
    'passwd',
    'name',
    'email',
    'tel',
    'register_dttm',
  )

