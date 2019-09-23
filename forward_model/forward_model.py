from collections import OrderedDict
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
import numpy as np
from copy import deepcopy

from simple_buffer import EpisodicBuffer
from parallel_module import ParallelModule
from normalizer import Normalizer
from utils import to_transitions, import_function

class ForwardModel(ParallelModule):
    def __init__(self, env_spec, buffer_size=int(1e6), lr=1e-4, hist_length=1, batch_size=64, network_params=None,
                 normalizer_params=None, train_steps=1, scope='forwardModel'):
        super().__init__(scope)

        self._env_spec = env_spec
        self._train_steps = train_steps
        self._lr = lr
        self._hist_length = hist_length
        self._batch_size = batch_size
        self._stage_shapes = OrderedDict(
            o=(self._env_spec['o_dim'],),
            a=(self._env_spec['a_dim'],),
            o_next=(self._env_spec['o_dim'],),
        )
        self._buffer = EpisodicBuffer(self._stage_shapes, buffer_size, self._env_spec['T'])

        self._network_params = network_params or dict()
        self._normalizer_params = normalizer_params or dict()

        self._sess = tf.get_default_session() or tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        ))
        with tf.variable_scope(self._scope, tf):
            self._staging_area_tf, self._input_ph_tf, self._staging_op_tf, self._batch_tf = self._prepare_staging()
            self._normalizer_tf = self._prepare_normalizers()
            self._net_tf = self._prepare_network()
            self._adam_tf, self._train_op_tf, self._loss_tf = self._prepare_training()

            self._sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.vars('')]
            self._sync_op_tf = [target.assign(source) for target, source in zip(self.vars(''), self._sync_source_tf)]

            self._global_sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.global_vars('')]
            self._global_sync_op_tf = [target.assign(source) for target, source in zip(self.global_vars(''), self._global_sync_source_tf)]

        self._sess.run(tf.variables_initializer(self.global_vars()))

    def _prepare_staging(self):
        with tf.variable_scope('staging', reuse=tf.AUTO_REUSE):
            staging_area_tf = StagingArea(
                dtypes=[tf.float32 for _ in self._stage_shapes.keys()],
                shapes=[(None,self._hist_length,*shape) for shape in self._stage_shapes.values()]
            )
            input_ph_tf = [tf.placeholder(tf.float32, shape=(None,self._hist_length,*shape))
                           for shape in self._stage_shapes.values()]
            staging_op_tf = staging_area_tf.put(input_ph_tf)

            batch_tf = OrderedDict([(key, batch_item)
                                    for key, batch_item in zip(self._stage_shapes.keys(), staging_area_tf.get())])

        return staging_area_tf, input_ph_tf, staging_op_tf, batch_tf

    def _prepare_normalizers(self):
        normalizer_tf = dict()
        with tf.variable_scope('o_stats'):
            normalizer_tf['o'] = Normalizer(self._env_spec['o_dim'], **self._normalizer_params)
        with tf.variable_scope('u_stats'):
            normalizer_tf['a'] = Normalizer(self._env_spec['a_dim'], **self._normalizer_params)
        with tf.variable_scope('abs_pred_err_stats'):
            normalizer_tf['abs_pred_err'] = Normalizer(self._env_spec['o_dim'], **self._normalizer_params)

        return normalizer_tf

    def _prepare_network(self):
        net_type = import_function(self._network_params['net_type'])
        self._logger.debug(f'Using {self._network_params["net_type"]}')
        params = deepcopy(self._network_params)
        del params['net_type']
        net = net_type(self._batch_tf, self._normalizer_tf, **params)
        return net

    def _prepare_training(self):
        adam_tf = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-5)
        loss_tf = tf.reduce_sum(tf.losses.get_losses(scope=self._scope))
        loss_tf += tf.losses.get_regularization_loss(scope=self._scope)
        train_op_tf = adam_tf.minimize(loss_tf, var_list=self.vars())

        return adam_tf, train_op_tf, loss_tf

    def store_transitions(self, batch, info):
        self._buffer.store_episode(batch)
        self._update_normalizers(batch)

    def _update_normalizers(self, batch):
        transitions = to_transitions(batch)
        self._normalizer_tf['o'].update(transitions['o'])
        self._normalizer_tf['a'].update(transitions['a'])

        self._normalizer_tf['o'].recompute_stats()
        self._normalizer_tf['a'].recompute_stats()

    def _update_pred_err_normalizers(self, batch):
        self._normalizer_tf['abs_pred_err'].update(np.abs(batch))

        self._normalizer_tf['abs_pred_err'].recompute_stats()

    def _stage_batch(self):
        batch = self._buffer.sample_transitions(self._batch_size, hist_length=self._hist_length)
        batch = [batch[key] for key in self._stage_shapes.keys()]

        self._sess.run(self._staging_op_tf, feed_dict=dict(zip(self._input_ph_tf, batch)))

    def train(self):
        losses = []
        for i in range(self._train_steps):
            self._stage_batch()
            loss, pred_errs, _ = self._sess.run([self._loss_tf, self._net_tf._prediction_error, self._train_op_tf])
            self._update_pred_err_normalizers(pred_errs)
            losses.append(loss)
        return {'loss': ('scalar', np.mean(losses))}

    def predict(self, o, a, o_next):
        pred = self._net_tf.predict(o, a, o_next)
        pred['abs_prediction_error_mean'] = self._sess.run(self._normalizer_tf['abs_pred_err'].mean)
        pred['abs_prediction_error_std'] = self._sess.run(self._normalizer_tf['abs_pred_err'].std)

        return pred

    def vars(self, scope=''):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)
        res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'o_stats/mean')
        res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'o_stats/std')
        res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'r_stats/mean')
        res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'r_stats/std')
        res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'abs_pred_err_stats/mean')
        res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'abs_pred_err_stats/std')
        assert len(res) > 0
        return res

    def global_vars(self, scope=''):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + scope)
        return res

    def get_params(self, scope=''):
        return (self._scope, [(str(var), self._sess.run(var)) for var in self.vars(scope)])

    def get_global_params(self, scope=''):
        return (self._scope, [(str(var), self._sess.run(var)) for var in self.global_vars(scope)])

    def set_params(self, params, scope=''):
        params = [param[1] for param in params]
        self._sess.run(self._sync_op_tf, feed_dict=dict([(ph, param) for ph, param in zip(self._sync_source_tf, params)]))

    def set_global_params(self, params, scope=''):
        params = [param[1] for param in params]
        self._sess.run(self._global_sync_op_tf, feed_dict=dict([(ph, param) for ph, param in zip(self._global_sync_source_tf, params)]))

if __name__ == '__main__':
    import dill
    from envs.continuous.BallPickUpExtendedObstacles import Ball

    tmp_env = Ball()
    obs = tmp_env.reset()
    env_spec = dict(
        o_dim=obs['observation'].shape[0],
        a_dim=tmp_env.action_space.shape[0],
        g_dim=obs['desired_goal'].shape[0],
        og_dim=obs['observation'].shape[0]+obs['desired_goal'].shape[0],
        T=800,
    )

    sess = tf.get_default_session() or tf.InteractiveSession()

    forward_model = ForwardModel(env_spec, hist_length=2)

    with open('/tmp/episode.pkl', 'rb') as f:
        epsisode, info = dill.load(f)

    forward_model.store_transitions(epsisode, info)
    print(sess.run([forward_model._normalizer_tf['o'].mean, forward_model._normalizer_tf['o'].std]))
    batch = forward_model._buffer.sample_transitions(forward_model._batch_size, hist_length=forward_model._hist_length)
    print(batch['o'][1])
    o = tf.placeholder(tf.float32, shape=batch['o'].shape)
    print(sess.run(forward_model._normalizer_tf['o'].normalize(o), feed_dict={o:batch['o']}))

    forward_model.train()

