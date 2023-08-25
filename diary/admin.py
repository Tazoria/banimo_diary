from django.contrib import admin
from .models import Diary


class DiaryAdmin(admin.ModelAdmin):
  search_fields = ['subject']
  list_display = ('idx', 'subject', 'writer', 'create_date')


admin.site.register(Diary, DiaryAdmin)
