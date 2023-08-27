from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required


# 로그인 함수
@require_http_methods(['GET', 'POST'])
def login(request):
  if request.user.is_authenticated:
    return redirect('diary:list')

  if request.method == 'POST':
    form = AuthenticationForm(request, request.POST)
    if form.is_valid():
      print('form is valid')
      auth_login(request, form.get_user())
      print(request.user)
      return redirect(request.GET.get('next') or 'diary:list')
  else:
    form = AuthenticationForm()
  context = {
    'form': form,
  }
  print('로그인이 안된 상태')
  return render(request, 'accounts/login.html', context)


def login_page(request):
  print(request.user)
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


@login_required
def logout(request):
    auth_logout(request)
    return redirect('/')
