from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Users
import hashlib


def home(request):
  return render(request, 'index.html')


# 로그인 함수
def login(request):
  if request.method == 'POST':
    userid = request.POST['userid']
    passwd = request.POST['passwd']
    passwd = hashlib.sha256(passwd.encode()).hexdigest()

    row = Users.objects.filter(userid=userid, passwd=passwd)
    if len(row) > 0:
      row = Users.objects.filter(userid=userid, passwd=passwd)[0]
      print(row)
      request.session['id'] = row.id
      request.session['name'] = row.name

      return redirect('/diary')
    else:
      return redirect('/')

  else:
    return redirect('/')


def join(request):
  if request.method == 'GET':
    return render(request, 'users/join.html')

  elif request.method == 'POST':
    userid = request.POST['userid']
    passwd = request.POST['passwd']
    passwd = hashlib.sha256(passwd.encode()).hexdigest()
    name = request.POST['name']
    email = request.POST['email']
    tel = request.POST['tel']

    Users(userid=userid, passwd=passwd, name=name, email=email, tel=tel).save()

    return render(request, 'users/login.html')


def logout(request):
  request.session.clear()

  return redirect('../')
