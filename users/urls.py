from django.urls import path
from . import views


app_name = 'users'

urlpatterns = [
  path('', views.home),
  path('join', views.join),
  path('login', views.login),
  path('logout', views.logout),
]