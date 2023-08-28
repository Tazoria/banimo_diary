import tensorflow as tf
import tensorflow_datasets as tfds
import re
import os


class Preprocessor:
  def __init__(self):
    self.tokenizer = None
    self.MAX_LENGTH = 43
    self.START_TOKEN = []
    self.END_TOKEN = []

  def seperate_punctuation_train(self, sentence_list):
    sentences = []
    for sentence in sentence_list:
      sentence = re.sub(r"([?.!,])", r" \1 ", sentence).strip()
      sentences.append(sentence)
      length = len(sentence.split())
      if length > self.MAX_LENGTH:
        self.MAX_LENGTH = length
    # print('reg_sentence: return(preprocessed) > ', sentences[:5])
    return sentences

  def get_tokenizer(self, sentence_list, target_vocab_size=2 ** 13):
    sentence_list = self.seperate_punctuation_train(sentence_list)
    self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      sentence_list, target_vocab_size=target_vocab_size)

    self.tokenizer.save_to_file(f'vocab_{target_vocab_size}')

    corpus = []
    vocab_path = f'vocab_{target_vocab_size}.subwords'
    with open(vocab_path, 'r', encoding='utf-8') as f:
      for inx, line in enumerate(f):
        if inx > 1:
          corpus.append(line.strip())
    # 헤더 정보가 포함된 열 두개를 토크나이저 로드시에 제외하기 때문에 추가해줌
    corpus.insert(0, '')
    corpus.insert(0, '')

    new_vocab_path = vocab_path.replace('.subwords', '.txt')
    with open(new_vocab_path, 'w', encoding='utf-8') as f:
      f.write('\n'.join(corpus))
      f.write('\n')
    os.remove(vocab_path)

    self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
    # print('get_tokenizer: return(self.tokenizer) > ', self.tokenizer)

    return self.tokenizer

  def tokenize_and_filter(self, inputs, outputs, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []
    self.tokenizer = tokenizer
    for (sentence1, sentence2) in zip(inputs, outputs):
      # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
      sentence1 = self.START_TOKEN + self.tokenizer.encode(sentence1) + self.END_TOKEN
      sentence2 = self.START_TOKEN + self.tokenizer.encode(sentence2) + self.END_TOKEN

      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)

    # 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=self.MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=self.MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs

  def get_train_dataset(self, questions, answers, batch_size=128, buffer_size=20000):
    dataset = tf.data.Dataset.from_tensor_slices((
      {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]  # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
      },
      {
        'outputs': answers[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
      },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
