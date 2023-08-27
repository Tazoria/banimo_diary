from django.forms import ModelForm, TextInput, Textarea
from diary.models import Diary, Comment


class DiaryForm(ModelForm):
    class Meta:
        model = Diary  # 사용할 모델
        fields = ['subject', 'content']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
        }
        widgets = {
            'subject': TextInput(attrs={
                'class': "form-control",
                'style': 'max-width: 300px;',
                'placeholder': '제목'
            }),
            'content': Textarea(attrs={
                'class': "form-control",
                'style': 'max-width: 300px;',
                'placeholder': '내용'
            }),
        }


class CommentForm(ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        labels = {
            'content': '답변내용',
        }