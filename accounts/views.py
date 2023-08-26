from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm


# 로그인 함수
def login(request):
  if request.user.is_authenticated:
    return redirect('diary:list')

  if request.method == 'POST':
    form = AuthenticationForm(request, request.POST)
    if form.is_valid():
      auth_login(request, form.get_user())
      print(request.user)
      return redirect(request.GET.get('next') or 'accounts:login')
  else:
    form = AuthenticationForm()
  context = {
    'form': form,
  }
  return render(request, 'index.html', context)


def login_page(request):
  return render(request, 'accounts/login.html')


def join(request):
  if request.method == 'POST':
    form = UserCreationForm(request.POST)
    if form.is_valid():
      form.save()
      return redirect('diary:list')
  else:
    form = UserCreationForm()
  context = {
    'form': form,
  }
  return render(request, 'accounts/join.html', context)


def logout(request):
    auth_logout(request)
    return redirect('/')