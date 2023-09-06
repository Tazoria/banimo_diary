import tensorflow_datasets as tfds


def load_tokenizer(vocab_path):
  corpus = []

  with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
      corpus.append(line.strip())
  tokenizer = tfds.deprecated.text.SubwordTextEncoder(vocab_list=corpus)
  return tokenizer


if __name__ == '__main__':
  vocab_path = '../keras/transformer/vocab_32000.txt'
  tokenizer = load_tokenizer(vocab_path)
  encoded = tokenizer.encode('안녕 나는 타조랑말코딱지야')
  decoded = tokenizer.decode(encoded)
  print(encoded)
  print(decoded)
