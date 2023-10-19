import tensorflow as tf
from trax.config import CONFIG


class SliceImage(tf.keras.layers.Layer):
  def __init__(self):
    super(SliceImage, self).__init__()
    self.min_size = CONFIG['MODEL_CONFIG']['SIZE_THRESHOLD']
    self.th = CONFIG['MODEL_CONFIG']['INTENSITY_THRESHOLD'] / 255.

  def call(self, inputs):
    inputs_shape = inputs.shape
    x = inputs
    col_lim = tf.argmin((tf.reduce_sum(tf.reduce_sum(inputs, axis=-1), axis=1) < self.th))
    row_lim = tf.argmin((tf.reduce_sum(tf.reduce_sum(inputs, axis=-1), axis=2) < self.th))
    if tf.reduce_all((col_lim > 0)):
      if tf.reduce_all(self.min_size > inputs_shape[1]-2*col_lim):
        col_lim = (inputs_shape[1]-self.min_size)//2
      x = x[:, col_lim:-col_lim, :]
    if tf.reduce_all((row_lim > 0)):
      if tf.reduce_all(self.min_size > inputs_shape[0]-2*row_lim):
        row_lim = (inputs_shape[0]-self.min_size)//2
      x = x[row_lim:-row_lim, :, :]
    tf.print(x.shape)
    return x
