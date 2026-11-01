import tensorflow as tf
from keras.layers import Layer

class attention(Layer):
    def __init__(self, return_sequences=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # Modern TensorFlow operations
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return output if self.return_sequences else tf.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape if self.return_sequences else (input_shape[0], input_shape[2])
