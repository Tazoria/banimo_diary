import time
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator
from diary.models import Diary
from diary.forms import DiaryForm
from models.utils.load_tokenizer import load_tokenizer
from models.transformer.transformer import transformer
from models.transformer.evaluate import Evaluate
import pandas as pd
import random

# import re
def get_bani_names(num_comments):
  banies = ['검은바니', '흰 바니', '분홍 바니', '무지개 바니', '노랑 바니',
            '초코 바니', '연두 바니', '모찌 바니', '회색 바니', '하늘 바니',
            '꼬마 바니', '방귀쟁이 바니', '장난꾸러기 바니', '재간둥이 바니', '꼬리가 긴 바니',
            '동글동글한 바니', '귀가 짧은 바니', '발이 작은 바니', '킁킁거리는 바니', '꽃향기 나는 바니', '아기 바니', '사랑둥이 바니', '어딘가 수상한 바니']

  # 생성된 댓글의 수만큼 바니의 이름을 랜덤 추출
  bani_names = random.sample(banies, num_comments)

  return bani_names

# r'D:\banimo_diary\models\data\bani_random_comments.xlsx'
def get_random_comments(num_sentences):
  random_comments = pd.read_excel('models/data/bani_random_comments.xlsx').fillna(0)
  # 결측치 제외한 나머지 값들을 담은 리스트 생성
  sentence = [word for word in random_comments['문장'].to_list() if word]
  punctuation = [word for word in random_comments['문장부호'].to_list() if word]

  # 트랜스포머 모델로부터 생성된 댓글의 수에 따라 단순 랜덤 댓글 생성
  #   5개 이하인 경우 2~4개
  #   5개 초과인 경우 1~2개
  if num_sentences <= 5:
    num_random_comments = random.randint(2, 4)
  else:
    num_random_comments = random.randint(1, 2)

  comments = []
  for i in range(num_random_comments):
    comment = random.choice(sentence) + ' ' + random.choice(punctuation)
    comments.append(comment)
  return comments


def get_bani_acts():
  bani_acts = pd.read_excel('models/data/bani_acts.xlsx').fillna(0)

  # 결측치 제외한 나머지 값들을 담은 리스트 생성
  place = [word for word in bani_acts['어디서'].to_list() if word]
  how = [word for word in bani_acts['어떻게'].to_list() if word]
  what = [word for word in bani_acts['무엇을'].to_list() if word]
  act1 = [word for word in bani_acts['행동1'].to_list() if word]
  act2 = [word for word in bani_acts['행동2'].to_list() if word]

  # 행동 정의
  act = '(바니가 당신에게로 와 ' + random.choice(place) + '에서 ' + random.choice(how) + ' '\
        + random.choice(act1) + ' ' + random.choice(what) + ' ' + random.choice(act2) + ' )'
  return [act]


def get_comment(content):
  # 저장된 트랜스포머 모델의 가중치를 적용하기 위해 단어사전과 모델 가중치의 경로 불러오기
  vocab_path = 'models/transformer/vocab_8192.txt'
  model_path = 'models/save/weights/transformer_weight150.h5'

  tokenizer = load_tokenizer(vocab_path)  # 토크나이저 생성

  model = transformer(vocab_size=tokenizer.vocab_size + 2,  # 단순 모델 정의
                      num_layers=2,
                      dff=512,
                      d_model=256,
                      num_heads=8,
                      dropout=.1)
  model.load_weights(model_path)  # 가중치 불러오기

  # 입력 받은 문단을 문장으로 자르기
  sentences = content.replace('\r', '-').replace('\n', '-').replace('.', '-')
  sentences = sentences.split('-')
  sentences = [sentence for sentence in sentences if sentence]
  print(sentences)

  if len(sentences) > 10:  # 문장 수가 10개가 넘으면 그 중에서 10개만 랜덤 추출
    sentences = random.sample(sentences, 10)

  comments_from_model = []  # 모델이 생성한 댓글을 저장할 리스트
  for sentence in sentences:
    evaluate = Evaluate(model, tokenizer)
    output = evaluate.predict(sentence.strip())  # 각 문장에 대한 댓글 생성
    comments_from_model.append(output)

  comments_from_model = list(set(comments_from_model))
  bani_acts = get_bani_acts()  # 바니의 행동을 랜덤으로 생성
  # 적절한 생성댓글의 수가 적은 경우를 대비해 긍정 의미를 갖는 단순 문장 생성
  random_comments = get_random_comments(len(comments_from_model))
  # 생성된 댓글의 수만큼 바니의 이름 갖고오기
  commenters = get_bani_names(len(comments_from_model) + len(bani_acts) + len(random_comments))

  # 생성된 댓글들의 합
  comments = {
    'commenter': commenters,
    'comment': comments_from_model + random_comments + bani_acts
  }

  return comments


def diary_list(request):
  if request.user.is_authenticated:
    page = request.GET.get('page', '1')
    diaries = Diary.objects.filter(writer=request.user.username).order_by('-create_date')

    paginator = Paginator(diaries, 10)
    page_obj = paginator.get_page(page)
    max_index = len(paginator.page_range)
    context = {'diaries': page_obj, 'max_index': max_index, 'page': page}

    return render(request, 'diary/list.html', context)
  else:
    return redirect('accounts:login')


def write(request):
  if request.user.is_authenticated:
    return render(request, 'diary/diary_form.html')
  else:
    return redirect('accounts:login')


def detail(request, diary_idx):
    diary = get_object_or_404(Diary, pk=diary_idx)
    return render(request, 'diary/detail.html',
                  {'diary': diary})


def insert(request):
  if request.method == 'POST':
    form = DiaryForm(request.POST)
    if form.is_valid():
      diary = form.save(commit=False)
      diary.writer = request.user
      diary.save()
      print(diary.subject, ' 저장 성공')

      comments = get_comment(request.POST['content'])
      for i in range(len(comments['comment'])):
        idx = diary.idx
        diary = get_object_or_404(Diary, pk=idx)
        diary.diary_idx.create(content=comments['comment'][i], commenter=comments['commenter'][i])

      return redirect('diary:list')
  else:
    return render(request, 'diary/diary_form.html')


# def update(request):
#   if request.method == 'POST':
#     form = DiaryForm(request.POST, instance=diary)
#     if form.is_valid():
#       diary = form.save(commit=False)
#       diary.modify_date = time.time()
#       diary.save()
#       print(diary.subject, ' 저장 성공')
#       return redirect('diary:detail', diary_idx=diary_idx)
#   else:
#     return render(request, 'diary/diary_form.html')
#     # diary = Diary(
#     #   idx=request.POST['idx'],
#     #   subject=request.POST['subject'],
#     #   writer=request.user.username,
#     #   create_date=request.POST['create_date'],
#     #   update_date=time.time.now(),
#     #   content=request.POST['content'])
#     # diary.save()


# def delete(request, diary_idx):
#   diary = get_object_or_404(Diary, pk=diary_idx)
#   if request.user != diary.author:
#     messages.error(request, '삭제권한이 없습니다')
#     return redirect('diary:detail', diary_idx=diary.idx)
#   diary.delete()
#   return redirect('diary:list')

