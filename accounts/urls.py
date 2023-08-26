from django.urls import path
from . import views


app_name = 'accounts'

urlpatterns = [
  path('login_page', views.login_page, name='login_page'),
  path('login', views.login, name='login'),
  path('logout', views.logout, name='logout'),
  path('join', views.join, name='join'),
]