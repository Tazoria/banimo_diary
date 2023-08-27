from django.shortcuts import render, redirect, get_object_or_404
from django.core.paginator import Paginator
from diary.models import Diary, Comment
from diary.forms import DiaryForm
# from models.utils.Preprocess import Preprocessor
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
            '동글동글한 바니', '귀가 짧은 바니', '발이 작은 바니', '킁킁거리는 바니', '꽃향기 나는 바니']

  bani_names = random.sample(banies, num_comments)

  return bani_names


def get_random_comments(num_sentences):
  random_comments = pd.read_excel(r'D:\banimo_diary\data\bani_random_sentences.xlsx')
  if num_sentences <= 5:
    num_random_comments = random.randint(2, 4)
  else:
    num_random_comments = random.randint(1, 2)

  comments = []
  for i in range(num_random_comments):
    comment = random.choice(random_comments['문장']) + ' ' + random.choice(random_comments['문장부호'])
    comments.append(comment)
  return comments


def get_bani_acts():
  bani_acts = pd.read_excel(r'D:\banimo_diary\data\bani_acts.xlsx').fillna(0)

  place = [word for word in bani_acts['어디서'].to_list() if word]
  how = [word for word in bani_acts['어떻게'].to_list() if word]
  what = [word for word in bani_acts['무엇을'].to_list() if word]
  act1 = [word for word in bani_acts['행동1'].to_list() if word]
  act2 = [word for word in bani_acts['행동2'].to_list() if word]

  act = '( ' + random.choice(place) + '에서 ' + random.choice(how) + ' ' + random.choice(act1) + ' ' + random.choice(
    what) + '를 ' + random.choice(act2) + ' )'
  return [act]


def get_comment(content):
  vocab_path = r'D:\banimo_diary\models\vocab.txt'
  model_path = r'D:\banimo_diary\models\save\weights\transformer_weight150.h5'
  tokenizer = load_tokenizer(vocab_path)

  model = transformer(vocab_size=tokenizer.vocab_size + 2,
                      num_layers=2,
                      dff=512,
                      d_model=256,
                      num_heads=8,
                      dropout=.1)
  model.load_weights(model_path)

  sentences = content.split('\n')
  if len(sentences) > 10:
    sentences = random.sample(sentences, 10)

  comments_from_model = []
  for sentence in sentences:
    evaluate = Evaluate(sentence.strip(), model, tokenizer)
    output = evaluate.predict()
    comments_from_model.append(output)

  bani_acts = get_bani_acts()
  random_comments = get_random_comments(len(comments_from_model))
  commenters = get_bani_names(len(comments_from_model) + len(bani_acts) + len(random_comments))

  comments = {
    'commenter': commenters,
    'comment': comments_from_model + random_comments + bani_acts
  }

  return comments


def diary_list(request):
  if request.user.is_authenticated:
    page = request.GET.get('page', '1')
    diaries = Diary.objects.order_by('create_date')

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
    # diary = Diary(
    #   subject=request.POST['subject'],
    #   writer=request.session['name'],
    #   content=request.POST['content'])
    # diary.save()
    form = DiaryForm(request.POST)
    if form.is_valid():
      diary = form.save(commit=False)
      diary.writer = request.user
      diary.save()

      print(diary.subject, ' 저장 성공')

      comments = get_comment(request.POST['content'])

      for i in range(len(comments['comment'])):
        # comment = Comment(
        #   diary_idx=diary,
        #   commenter=comments['commenter'][i],
        #   content=comments['comment'][i],
        # )
        # comment.save()
        idx = diary.idx
        diary = get_object_or_404(Diary, pk=idx)
        diary.diary_idx.create(content=comments['comment'][i], commenter=comments['commenter'][i])
      return redirect('diary:list')
  else:
    return render(request, 'diary/diary_form.html')


def update(request):
  diary = Diary(
    idx=request.POST['idx'],
    subject=request.POST['subject'],
    writer=request.POST['name'],
    create_date=request.POST['create_date'],
    content=request.POST['content'])
  diary.save()
  get_comment(request)
  return redirect('/detail')


def delete(request, diary_idx):
  Diary.objects.get(diary_idx=request.POST['idx']).delete()
  return redirect('/diary')

