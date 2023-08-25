import tensorflow as tf
import re
import tensorflow_datasets as tfds
from transformer_model import *
sentence = '''오늘 하루 너무 힘든 거 같아요. 공부도 잘 안되고 나태하기 짝이 없어요.
어떻게 하면 좋을까요? 너무 힘이들어요.!..
빨리 취업을 해야하는데 이러면 못 할 것 같아요
뭘 해야하는지 알지만 몸과 마음이 안따라주니 너무 힘들어요.
저도 얼른 취업해서 행복해지고 싶어요.
남들처럼 평범하게 살고싶어요..'''

MAX_LENGTH = 42
corpus = []
with open('../../vocab.txt', 'r', encoding='utf-8') as f:
    for line in f:
        corpus.append(line.strip())
# tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
#     sentence, target_vocab_size=2**13)
# tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('vocab.txt')
tokenizer = tfds.deprecated.text.SubwordTextEncoder(vocab_list=corpus)
# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
# VOCAB_SIZE = tokenizer.vocab_size + 2
VOCAB_SIZE = tokenizer.vocab_size + 2
# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2

tf.keras.backend.clear_session()

# 하이퍼파라미터
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

print('===================================')
model.load_weights(r"D:\banimo_diary\models\checkpoints\weights\model_sample.h5")
def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence


def evaluate(sentence):
  # 입력 문장에 대한 전처리
  sentence = preprocess_sentence(sentence)

  # 입력 문장에 시작 토큰과 종료 토큰을 추가
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)
  print(model.summary())
  # 디코더의 예측 시작
  for i in range(MAX_LENGTH):
    print('-------------------------')
    print(sentence, output, sep='\n')
    predictions = model(inputs=[sentence, output], training=False)
    print('=======================================')

    # 현재 시점의 예측 단어를 받아온다.
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 현재 시점의 예측 단어가 종료 토큰이라면 예측을 중단
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 현재 시점의 예측 단어를 output(출력)에 연결한다.
    # output은 for문의 다음 루프에서 디코더의 입력이 된다.
    output = tf.concat([output, predicted_id], axis=-1)

  # 단어 예측이 모두 끝났다면 output을 리턴.
  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)

  # prediction == 디코더가 리턴한 챗봇의 대답에 해당하는 정수 시퀀스
  # tokenizer.decode()를 통해 정수 시퀀스를 문자열로 디코딩.
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])
  predicted_sentence = predicted_sentence.replace("'", '').replace('_', ' ')
  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence


# print(preprocess_sentence(sentence))
# print(evaluate(sentence))
# # print(predict(sentence))

output = predict(sentence)