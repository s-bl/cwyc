import numpy as np
import tensorflow as tf
from collections import OrderedDict
from copy import deepcopy

from policies.sac.value_function import NNQFunction, NNVFunction
from policies.sac.gaussian_policy import GaussianPolicy
from simple_buffer import SimpleBuffer
from parallel_module import ParallelModule
from task.utils import mg_fn


class SAC(ParallelModule):
    def __init__(self, env_spec, task_spec, scope='policy', buffer_size=int(1e6), batch_size=64, lr=3e-3, scale_reward=1,
                 discount=0.99, tau=0.01, target_update_interval=1, action_prior='uniform', reparameterize=False,
                 train_steps=1, q_hidden=(256, 256), v_hidden=(256, 256), pi_hidden=(256, 256), relative_goals=False):
        super().__init__(scope=scope)

        self._task_spec = task_spec

        self._relative_goals = relative_goals

        self._env_specs = deepcopy(env_spec)
        self._env_specs['og_dim'] = self._env_specs['o_dim']+self._task_spec['g_dim']
        self._stage_shapes = OrderedDict(
            o=(env_spec['o_dim'],),
            o_next=(env_spec['o_dim'],),
            a=(env_spec['a_dim'],),
            ag=(self._task_spec['g_dim'],),
            ag_next=(self._task_spec['g_dim'],),
            g=(self._task_spec['g_dim'],),
            g_next=(self._task_spec['g_dim'],),
            r=(1,),
            d=(1,),
        )
        self._buffer = SimpleBuffer(self._stage_shapes, buffer_size)

        self._batch_size = batch_size
        self.do_train = True

        self._policy_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._scale_reward = scale_reward
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._q_hidden = q_hidden
        self._pi_hidden = pi_hidden
        self._v_hidden = v_hidden

        self._sess = tf.get_default_session() or tf.get_default_session() or tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        ))

        # Reparameterize parameter must match between the algorithm and the
        # policy actions are sampled from.
        self._reparameterize = reparameterize

        self._training_ops = list()

        self._train_steps = train_steps

        with tf.variable_scope(self._scope):
            self._prep_nets()
            self._init_placeholders()
            self._init_actor_update()
            self._init_critic_update()
            self._init_target_ops()

            self._sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.vars('')]
            self._sync_op_tf = [target.assign(source) for target, source in zip(self.vars(''), self._sync_source_tf)]

            self._global_sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.global_vars('')]
            self._global_sync_op_tf = [target.assign(source) for target, source in zip(self.global_vars(''), self._global_sync_source_tf)]

        self._sess.run(tf.variables_initializer(self.global_vars('')))

        self._logger.debug(f'Q_hidden: {q_hidden}, V_hidden: {v_hidden}, Pi_hidden: {pi_hidden}')

    def _mg_fn(self, x):
        return mg_fn(x, self._task_spec['mg'])

    def _prep_nets(self):
        self._qf1_net = NNQFunction(env_spec=self._env_specs, hidden_layer_sizes=self._q_hidden, name='qf1')
        self._qf2_net = NNQFunction(env_spec=self._env_specs, hidden_layer_sizes=self._q_hidden, name='qf2')
        self._vf_net = NNVFunction(env_spec=self._env_specs, hidden_layer_sizes=self._v_hidden)
        self._policy_net = GaussianPolicy(
            env_spec=self._env_specs,
            hidden_layer_sizes=self._pi_hidden,
            reparameterize=self._reparameterize,
            reg=1e-3,
        )

    def store_transitions(self, batch, info):
        # Get all transitions collected while in self._task.id
        new_batch = {key: np.vstack([np.asarray(value[i][np.squeeze(batch['tasks'][i]) == self._task_spec['id']]) for i in range(len(value))])
                             for key, value in batch.items()}


        batch_sizes = [value.shape[0] for value in new_batch.values()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        if batch_size == 0: return

        new_batch['ag'] = self._mg_fn(new_batch['ag'])
        new_batch['ag_next'] = self._mg_fn(new_batch['ag_next'])
        new_batch['g'] = self._mg_fn(new_batch['g'])
        new_batch['g_next'] = self._mg_fn(new_batch['g_next'])

        self._buffer.store_transitions(deepcopy(new_batch))

    def _sample_batch(self):
        transitions = self._buffer.sample(self._batch_size)

        if self._relative_goals:
            g = transitions['g'] - transitions['ag']
            g_next = transitions['g_next'] - transitions['ag_next']
        else:
            g = transitions['g']
            g_next = transitions['g_next']

        batch = {
            'o': np.hstack([transitions['o'], g]),
            'a': transitions['a'],
            'r': np.squeeze(transitions['r']),
            'd': np.squeeze(transitions['d']),
            'o_next': np.hstack([transitions['o_next'], g_next]),
        }

        return batch

    def train(self):
        """Initiate training of the SAC instance."""
        if self._buffer.get_current_size() == 0: return

        if not self.do_train: return

        for i in range(self._train_steps):
            batch = self._sample_batch()
            self._do_training(batch)

    def get_actions(self, o, ag, g, *args, **kwargs):
        if self._relative_goals:
            g = self._mg_fn(g) - self._mg_fn(ag)
        else:
            g = self._mg_fn(g)
        observations = np.hstack([o, g])
        ret = self._policy_net.get_actions(observations)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def vars(self, scope=''):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)
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

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._env_specs['og_dim']),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._env_specs['og_dim']),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._env_specs['a_dim']),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    @property
    def scale_reward(self):
        return self._scale_reward

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """

        self._qf1_tf = self._qf1_net.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N
        self._qf2_tf = self._qf2_net.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N

        with tf.variable_scope('target'):
            vf_next_target_tf = self._vf_net.get_output_for(self._next_observations_ph)  # N
            self._vf_target_params_tf = self._vf_net.get_params_internal()

        ys = tf.stop_gradient(
            self.scale_reward * self._rewards_ph +
            (1 - self._terminals_ph) * self._discount * vf_next_target_tf
        )  # N

        self._td_loss1_tf = 0.5 * tf.reduce_mean((ys - self._qf1_tf)**2)
        self._td_loss2_tf = 0.5 * tf.reduce_mean((ys - self._qf2_tf)**2)

        qf1_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss1_tf,
            var_list=self._qf1_net.get_params_internal()
        )
        qf2_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss2_tf,
            var_list=self._qf2_net.get_params_internal()
        )

        self._training_ops.append(qf1_train_op)
        self._training_ops.append(qf2_train_op)

    def _init_actor_update(self):
        """Create minimization operations for policy and state value functions.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and value functions with gradient descent, and appends them to
        `self._training_ops` attribute.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        See Equations (8, 13) in [1], for further information
        of the value function and policy function update rules.
        """

        actions, log_pi = self._policy_net.actions_for(observations=self._observations_ph,
                                                       with_log_pis=True)

        self._vf_tf = self._vf_net.get_output_for(self._observations_ph, reuse=True)  # N
        self._vf_params_tf = self._vf_net.get_params_internal()

        if self._action_prior == 'normal':
            D_s = actions.shape.as_list()[-1]
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(D_s), scale_diag=tf.ones(D_s))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        log_target1 = self._qf1_net.get_output_for(
            self._observations_ph, actions, reuse=True)  # N
        log_target2 = self._qf2_net.get_output_for(
            self._observations_ph, actions, reuse=True)  # N
        min_log_target = tf.minimum(log_target1, log_target2)

        if self._reparameterize:
            policy_kl_loss = tf.reduce_mean(log_pi - log_target1)
        else:
            policy_kl_loss = tf.reduce_mean(log_pi * tf.stop_gradient(
                log_pi - log_target1 + self._vf_tf - policy_prior_log_probs))

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy_net.name)
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)

        policy_loss = (policy_kl_loss
                       + policy_regularization_loss)

        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        self._vf_loss_tf = 0.5 * tf.reduce_mean((
          self._vf_tf
          - tf.stop_gradient(min_log_target - log_pi + policy_prior_log_probs)
        )**2)

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=policy_loss,
            var_list=self._policy_net.get_params_internal()
        )

        vf_train_op = tf.train.AdamOptimizer(self._vf_lr).minimize(
            loss=self._vf_loss_tf,
            var_list=self._vf_params_tf
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)

    def _init_target_ops(self):
        """Create tensorflow operations for updating target value function."""

        source_params = self._vf_params_tf
        target_params = self._vf_target_params_tf

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    def _init_training(self, env, policy, buffer):
        self._sess.run(self._target_ops)

    def _do_training(self, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feed_dict)

        self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['o'],
            self._actions_ph: batch['a'],
            self._next_observations_ph: batch['o_next'],
            self._rewards_ph: batch['r'],
            self._terminals_ph: batch['d'],
        }

        return feed_dict

    # def logs(self, prefix=''):
    #     if self._buffer.current_size == 0:
    #         return []
    #
    #     feed_dict = self._get_feed_dict(self._sample_batch())
    #     qf1, qf2, vf, td_loss1, td_loss2 = self._sess.run(
    #         (self._qf1_tf, self._qf2_tf, self._vf_tf, self._td_loss1_tf, self._td_loss2_tf), feed_dict)
    #
    #     logs = []
    #     logs += [('qf1-avg', np.mean(qf1))]
    #     logs += [('qf1-std', np.std(qf1))]
    #     logs += [('qf2-avg', np.mean(qf1))]
    #     logs += [('qf2-std', np.std(qf1))]
    #     logs += [('mean-qf-diff', np.mean(np.abs(qf1-qf2)))]
    #     logs += [('vf-avg', np.mean(vf))]
    #     logs += [('vf-std', np.std(vf))]
    #     logs += [('mean-sq-bellman-error1', td_loss1)]
    #     logs += [('mean-sq-bellman-error2', td_loss2)]
    #
    #     if prefix is not '' and not prefix.endswith('/'):
    #         return [(prefix + '/' + key, val) for key, val in logs]
    #     else:
    #         return logs

    # def __getstate__(self):
    #     """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
    #     """
    #
    #     state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in self._excluded_subnames])}
    #     return state
    #
    # def __setstate__(self, state):
    #
    #     self.__dict__.update(state)
    #     self._sess = tf.get_default_session() or tf.InteractiveSession()
    #
    #     if not hasattr(self, '_q_hidden'):
    #         self._q_hidden = (self._layer_size, self._layer_size)
    #         self._pi_hidden = (self._layer_size, self._layer_size)
    #         self._v_hidden = (self._layer_size, self._layer_size)
    #
    #     self._training_ops = list()
    #     with tf.variable_scope(self.scope):
    #         self._prep_nets()
    #         self._init_placeholders()
    #         self._init_actor_update()
    #         self._init_critic_update()
    #         self._init_target_ops()

if __name__ == '__main__':
    from config import basic_configure

    config = basic_configure()

    policies = [policy_fn[0](**policy_fn[1]) for policy_fn in config['policies_fn']]
