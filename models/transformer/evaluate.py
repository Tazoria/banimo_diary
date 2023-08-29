from models.transformer import transformer
from models.utils.load_tokenizer import load_tokenizer
import tensorflow as tf
import re


class Evaluate:
  def __init__(self, model, tokenizer):
    self.sentence = ''
    self.tokenizer = tokenizer
    self.model = model

  def preprocess_sentence(self, sentence):
    # 단어와 구두점 사이에 공백 추가.
    # ex) 12시 땡! -> 12시 땡 !
    self.sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    self.sentence = self.sentence.strip()

    return self.sentence

  def evaluate(self, sentence):
    # 입력 문장에 대한 전처리
    self.sentence = self.preprocess_sentence(sentence)

    START_TOKEN, END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]

    # 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
    VOCAB_SIZE = self.tokenizer.vocab_size + 2

    # 입력 문장에 시작 토큰과 종료 토큰을 추가
    self.sentence = tf.expand_dims(
        START_TOKEN + self.tokenizer.encode(self.sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    MAX_LENGTH = 43
    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
      predictions = self.model(inputs=[self.sentence, output], training=False)
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

  def predict(self, sentence):
    prediction = self.evaluate(sentence)
    # prediction == 디코더가 리턴한 챗봇의 대답에 해당하는 정수 시퀀스
    # tokenizer.decode()를 통해 정수 시퀀스를 문자열로 디코딩.
    predicted_sentence = self.tokenizer.decode(
        [i for i in prediction if i < self.tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence.replace("'", '').replace('_', ' ')


if __name__ == '__main__':
  # vocab_path = r'D:\banimo_diary\models\vocab.txt'
  vocab_path = r'../vocab.txt'
  sample_sentence = '오늘 공부가 잘 안됐어. 그래서 신나게 놀았어. 참 재미있었어'
  model_path = r'../save/weights/transformer_weight150.h5'

  tokenizer = load_tokenizer(vocab_path)
  model = transformer.transformer(
    vocab_size=tokenizer.vocab_size + 2,
    num_layers=2,
    dff=512,
    d_model=256,
    num_heads=8,
    dropout=.1)
  model.load_weights(model_path)

  evaluate = Evaluate(model, tokenizer)
  output = evaluate.predict(sample_sentence)

  print('input > ', sample_sentence)
  print('output > ', output)
