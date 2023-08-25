import tensorflow_datasets as tfds


def load_tokenizer(vocab_path):
  corpus = []

  with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
      corpus.append(line.strip())
  tokenizer = tfds.deprecated.text.SubwordTextEncoder(vocab_list=corpus)

  return tokenizer
