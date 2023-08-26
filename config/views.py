from django.shortcuts import render, redirect


def home(request):
  print(request.user)
  return render(request, 'index.html')
