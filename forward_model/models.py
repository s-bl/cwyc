import tensorflow as tf
import numpy as np

from utils import import_function

class ForwardModelMLPStateDiff:
    def __init__(self, inputs, normalizer, nL=(255,255), activation='tensorflow.nn:tanh', layer_norm=False, scope='mlp'):
        self._sess = tf.get_default_session() or tf.InteractiveSession()

        activation = import_function(activation)

        self._o = inputs['o']
        self._a = inputs['a']
        self._o_next = inputs['o_next'][:,-1]
        self._layer_norm = layer_norm

        o_norm = normalizer['o'].normalize(self._o)
        a_norm = normalizer['a'].normalize(self._a)
        o_next_norm = normalizer['o'].normalize(self._o_next)
        state_diff = o_next_norm - o_norm[:,-1]

        o_norm_flat = tf.reshape(o_norm, (-1, np.prod(o_norm.shape[1:])))
        a_norm_flat = tf.reshape(a_norm, (-1, np.prod(a_norm.shape[1:])))

        l = tf.concat([o_norm_flat, a_norm_flat], axis=1)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for i, n in enumerate(nL):
                l = tf.layers.dense(l, n, name=f'layer{i}')
                if self._layer_norm:
                    l = tf.contrib.layers.layer_norm(l)
                l = activation(l)

            self._predicted_state_diff = tf.layers.dense(l, self._o_next.shape[1], name='final_layer')

        self._predicted_state = normalizer['o'].denormalize(o_norm[:,-1] + self._predicted_state_diff)
        self._prediction_error = self._o_next - self._predicted_state

        self._loss = tf.losses.mean_squared_error(state_diff, self._predicted_state_diff)

    def predict(self, o, a, o_next):
        return (self._sess.run(dict(
            predicted_state_diff=self._predicted_state_diff,
            predicted_state=self._predicted_state,
            prediction_error=self._prediction_error
        ),
            feed_dict={
                self._o:o,
                self._a:a,
                self._o_next:o_next
            }
        ))

class ForwardModelMLP:
    def __init__(self, inputs, normalizer, nL=(255,255), layer_norm=False, scope='mlp'):
        self._sess = tf.get_default_session() or tf.InteractiveSession()

        self._o = inputs['o']
        self._a = inputs['a']
        self._o_next = inputs['o_next'][:,-1]
        self._layer_norm = layer_norm

        o_norm = normalizer['o'].normalize(self._o)
        a_norm = normalizer['a'].normalize(self._a)
        o_next_norm = normalizer['o'].normalize(self._o_next)

        o_norm_flat = tf.reshape(o_norm, (-1, np.prod(o_norm.shape[1:])))
        a_norm_flat = tf.reshape(a_norm, (-1, np.prod(a_norm.shape[1:])))

        l = tf.concat([o_norm_flat, a_norm_flat], axis=1)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for i, n in enumerate(nL):
                l = tf.layers.dense(l, n, name=f'layer{i}')
                if self._layer_norm:
                    l = tf.contrib.layers.layer_norm(l)
                l = tf.nn.relu(l)

            self._predicted_state_norm = tf.layers.dense(l, self._o_next.shape[1], name='final_layer')

        self._predicted_state = normalizer['o'].denormalize(self._predicted_state_norm)
        self._prediction_error = self._o_next - self._predicted_state

        self._loss = tf.losses.mean_squared_error(o_next_norm, self._predicted_state_norm)

    def predict(self, o, a, o_next):
        return (self._sess.run(dict(
            predicted_state=self._predicted_state,
            prediction_error=self._prediction_error
        ),
            feed_dict={
                self._o:o,
                self._a:a,
                self._o_next:o_next
            }
        ))





