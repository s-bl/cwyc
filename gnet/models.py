import os
import numpy as np
import tensorflow as tf
from scipy.special import binom
from scipy.linalg import pinv
from task.utils import mg_fn
from tensorboardX import SummaryWriter
import logging
import matplotlib.pyplot as plt

from gnet.utils import plot_dense_weights
from utils import get_jobdir

class GnetModel:
    def __init__(self, parent, inputs, normalizer, tasks_specs, task_from_id, task_to_id, scope):
        self._parent = parent
        self._scope = scope
        self._normalizer = normalizer
        self._sess = tf.get_default_session() or tf.InteractiveSession()
        self._inputs = inputs
        self._tasks_specs = tasks_specs
        self._task_to_id = task_to_id
        self._task_from_id = task_from_id

    def sample_goal(self, o, constant_coords, variable_coords):
        raise NotImplementedError

    def write_tflogs(self, scope=''):
        pass

class GnetPairwiseRelations(GnetModel):
    def __init__(self, parent, inputs, normalizer, tasks_specs, task_from_id, task_to_id, excluded_coords=None, use_bias=True, beta_trainable=True, beta_init=0, beta_clip=[-2, 3],
                 noise_stddev=0.001, l1=0, l2=0, scope='pairwise_relations'):

        super().__init__(parent=parent, inputs=inputs, normalizer=normalizer, tasks_specs=tasks_specs,
                         task_from_id=task_from_id, task_to_id=task_to_id, scope=scope)

        self._o = inputs['o']
        self._r = inputs['r']

        normalized_o = self._normalizer['o'].normalize(self._o)

        self._o_inp = normalized_o

        n = self._o_inp.shape[1]
        width = binom(int(n), 2).astype(np.int32)

        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):

            self._w_raw = tf.Variable(name='w_raw', initial_value=np.zeros((n, width)), dtype=tf.float32, trainable=True)


            self._w_noise = tf.truncated_normal([int(n), width], stddev=noise_stddev, seed=self._parent._seed)
            self._w_w_noise = self._w_raw + self._w_noise

            self._mask = np.zeros((n, width))

            l = 0
            for k in range(n - 1):
                j = k + 1
                for i in range(n - 1 - k):
                    self._mask[k, l] = 1 if excluded_coords is None or k not in excluded_coords and j not in excluded_coords else 0
                    self._mask[j, l] = 1 if excluded_coords is None or k not in excluded_coords and j not in excluded_coords else 0
                    j += 1
                    l += 1

            self._W = self._w_w_noise * self._mask
            self._b = tf.Variable(name='bias', initial_value=[0] * width, trainable=use_bias, dtype=tf.float32)


            self._hidden = tf.matmul(self._o_inp, self._W) + self._b

            self._beta = tf.Variable(name='beta', initial_value=beta_init, dtype=tf.float32,
                                    trainable=beta_trainable)


            self._beta_cliped = tf.clip_by_value(self._beta, *beta_clip)

            self._beta_cliped_exp = tf.exp(self._beta_cliped)

            self._output = tf.exp(-1 / 2 * tf.reduce_sum(tf.square(self._hidden), axis=1) * self._beta_cliped_exp)
            self._output = tf.reshape(self._output, (-1, 1))

            # loss

            self._loss = tf.losses.mean_squared_error(self._r, self._output)

            # Reg losses

            w_raw_l1_loss = tf.reduce_sum(tf.abs(self._w_raw)) * l1
            w_raw_l2_loss = tf.reduce_sum(tf.square(self._w_raw)) * l2

            tf.losses.add_loss(w_raw_l1_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
            tf.losses.add_loss(w_raw_l2_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

            b_l1_b_loss = tf.reduce_sum(tf.abs(self._b)) * l1
            b_l2_b_loss = tf.reduce_sum(tf.square(self._b)) * l2

            tf.losses.add_loss(b_l1_b_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
            tf.losses.add_loss(b_l2_b_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

            beta_l2_beta_loss = tf.reduce_sum(tf.square(self._beta)) * l2

            tf.losses.add_loss(beta_l2_beta_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

    def sample_goal(self, o, constant_coords, variable_coords):
        inverse = self._inverse(o, constant_coords, variable_coords)
        goal = np.asarray([inverse[i] for i in range(len(variable_coords))
                if variable_coords[i] in self._tasks_specs[self._task_from_id]['mo']])

        return goal

    def _inverse(self, x, constant_coords, variable_coords):
        """
            Calculate the inverse under conditions of fixed dimensions in input.
            update_numpy_params should be called before
        """
        params = self.get_params()
        W = params['W']
        b = params['b']

        Q1 = W[constant_coords]
        Q2 = W[variable_coords]

        fixed_contrib = np.matmul(x[constant_coords], Q1)

        Q2_ = pinv(Q2)

        x = np.matmul(-fixed_contrib - b, Q2_)

        return x

    def get_params(self):
        """
            Function to occassionally extract numpy parameters
            to avoid the overhead of a session call.
        """
        return self._sess.run({'W': self._W, 'b': self._b, 'beta_exp': self._beta_cliped_exp})

    def write_tflogs(self, scope=''):
        if not hasattr(self, '_summary_writer'):
            jobdir = get_jobdir(logging.getLoggerClass())
            self._summary_writer = SummaryWriter(os.path.join(jobdir, 'tf_board', scope))
            self._step = tf.get_variable(self._scope + '/step', shape=None, dtype=tf.float32, trainable=False, initializer=0.)
            self._sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/step')))
            self._step_inc_op = self._step.assign(tf.add(self._step, 1.))

        params = self.get_params()
        W = params['W']
        b = params['b']
        beta_exp = params['beta_exp']

        current_step = int(self._sess.run(self._step))

        if current_step % 10 == 0:
            fig = plot_dense_weights(W, b)
            self._summary_writer.add_figure(scope + '/weights', fig, current_step)
            plt.close(fig)
        self._summary_writer.add_scalar(scope + f'/beta_exp', beta_exp, current_step)

        self._sess.run(self._step_inc_op)

class Simple(GnetModel):
    def __init__(self, parent, inputs, normalizer, tasks_specs, task_from_id, task_to_id, scope='simple'):
        super().__init__(parent=parent, inputs=inputs, normalizer=normalizer, tasks_specs=tasks_specs,
                         task_from_id=task_from_id, task_to_id=task_to_id, scope=scope)

    def sample_goal(self, o, constant_coords, variable_coords):
        return mg_fn(o, self._tasks_specs[self._task_to_id]['mg'])

