import tensorflow as tf
from models.keras.transformer.transformer import transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  def get_config(self):
    config = {
      'd_model': self.d_model,
      'warmup_steps': self.warmup_steps,
    }
    return config


def loss_function(y_true, y_pred, max_length=43):
  y_true = tf.reshape(y_true, shape=(-1, max_length-1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


def accuracy(y_true, y_pred, max_length=43):
  # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, max_length-1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def get_model(vocab_size, num_layers=2, dff=512, d_model=256, num_heads=8, dropout=0.1):
  model = transformer(
    vocab_size=vocab_size,
    num_layers=num_layers,
    dff=dff,
    d_model=d_model,
    num_heads=num_heads,
    dropout=dropout)

  learning_rate = CustomSchedule(d_model)
  optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
  return model


def train_model_simple(model, dataset, epochs=50, mini_epoch=10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  total_epoch = 0
  tf.keras.backend.clear_session()

  for e in range(int(epochs / mini_epoch)):
    model.fit(dataset, epochs=mini_epoch, workers=-1, callbacks=[early_stopping])
    total_epoch += mini_epoch

  model.save('model.h5')

  return model
