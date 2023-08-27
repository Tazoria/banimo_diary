from django.urls import path
from diary import views

app_name = 'diary'

urlpatterns = [
    path('', views.diary_list, name='list'),
    path('<int:diary_idx>/', views.detail, name='detail'),
    path('write', views.write, name='write'),
    path('insert', views.insert, name='insert'),
    path('update', views.update, name='update'),
    path('delete', views.delete, name='delete'),
]