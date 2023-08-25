import tensorflow as tf
from transformer_model import *
import keras.optimizers
import os
import argparse

'''
[필요한 것]
- get_model: VOCAB_SIZE
- get_crit
- get_optimizer
- get_scheduler
- evaluate
- def main: 위의 것을 모두 포함해 실행시킬 함수 -> if로  __main__ 확인해서 돌리는 부분에 넣으면 됨
'''

# def define_argparser(is_continue=False):
#   p = argparse.ArgumentParser()
#   if is_continue:
#     p.add_argument(
#       '--weights',
#       required=True,
#       help='Model file name to continue')
#
#   p.add_argument(
#     '--train',
#     required=True,
#     help='Training set file name'
#   )


class Train:
  def __init__(self, dataset, epochs, mini_batch):
    self.dataset = dataset
    self.epochs = 100
    self.mini_batch = 10

    self.model = None

  def get_model(self):

    model_path = f'save/weights/transformer_weight{self.total_epoch}.h5'
    sample_model_path = f'save/weights/model_sample.h5'

    model = transformer(
      vocab_size=8249,
      num_layers=2,
      dff=512,
      d_model=256,
      num_heads=8,
      dropout=.1)

    # 학습된 최근 모델의 가중치 불러오기
    if os.path.isfile(model_path):
      model.load_weights(model_path)
      print(f'최근 학습모델 가중치 로드 완료')
    else:
      print('저장된 가중치가 없어 처음부터 학습 시작')
    self.model = model
    return model

  def train(self):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    epoch_saved_path = './save/weights/epoch_saved.txt'

    for e in range(int(self.epochs / self.mini_epoch)):
      self.model.fit(self.dataset, epochs=self.mini_epoch, workers=-1, callbacks=[early_stopping])
      total_epoch += mini_epoch
      with open(self.epoch_saved_path, 'w', encoding='utf-8') as f:
        f.write(str(total_epoch))

      # 5 epochs마다 모델 저장
      if total_epoch % 5 == 0:
        self.model.save_weights(f'save/weights/transformer_weight{total_epoch}.h5')

# def loss_function(y_true, y_pred):
#   y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
#
#   loss = tf.keras.losses.SparseCategoricalCrossentropy(
#       from_logits=True, reduction='none')(y_true, y_pred)
#
#   mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
#   loss = tf.multiply(loss, mask)
#
#   return tf.reduce_mean(loss)
#
#
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#         self.warmup_steps = warmup_steps
#
#     def get_config(self):
#         config = {
#           'd_model': self.d_model,
#           'warmup_steps': self.warmup_steps,
#         }
#         return config
#
#     def __call__(self, step):
#         step = tf.cast(step, tf.float32)  ###
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps**-1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)




if __name__ == '__main__':
  tf.keras.backend.clear_session()

