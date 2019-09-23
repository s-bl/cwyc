from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from utils import import_function, to_transitions
from normalizer import Normalizer
from policies.her.replay_buffer import ReplayBuffer
from policies.her.her_sampler import make_sample_her_transitions
from task.task import Task
from policies.her.util import simple_goal_subtract, flatten_grads, transitions_in_episode_batch
from parallel_module import ParallelModule
from policies.her.noise import get_noise_from_string
from policies.const import TRAIN, EVAL
from policies.her.mpi_adam import MpiAdam

from copy import deepcopy

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

class DDPG(ParallelModule):
    def __init__(self, env_spec, task_spec, buffer_size, network_params, normalizer_params, polyak, batch_size,
                 Q_lr, pi_lr, max_u, action_l2, clip_obs, scope, random_eps, noise_eps, train_steps,
                 relative_goals, clip_pos_returns, clip_return, replay_strategy, replay_k,
                 noise_type, share_experience, noise_adaptation, reuse=False):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
            bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
            q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            num_demo: Number of episodes in to be used in the demonstration buffer
            demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            prm_loss_weight: Weight corresponding to the primary loss
            aux_loss_weight: Weight corresponding to the auxilliary loss also called the cloning loss
        """
        super().__init__(scope)
        self.replay_k = replay_k
        self.replay_strategy = replay_strategy
        self.clip_pos_returns = clip_pos_returns
        self.relative_goals = relative_goals
        self.train_steps = train_steps
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.clip_obs = clip_obs
        self.action_l2 = action_l2
        self.max_u = max_u
        self.pi_lr = pi_lr
        self.Q_lr = Q_lr
        self.batch_size = batch_size
        self.normalizer_params = normalizer_params
        self.polyak = polyak
        self.buffer_size = buffer_size
        self._env_spec = env_spec
        self._T = self._env_spec['T']
        self._task_spec = task_spec
        self.network_params = network_params
        self._share_experience = share_experience
        self._noise_adaptation = noise_adaptation

        self._task_spec = deepcopy(task_spec)
        self._task_spec['buffer_size'] = 0
        self._task = Task(**self._task_spec)

        self._gamma = 1. - 1. / self._T
        self.clip_return = (1. / (1. - self._gamma)) if clip_return else np.inf

        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(network_params['net_type'])

        self.input_dims = dict(
            o=self._env_spec['o_dim'],
            a=self._env_spec['a_dim'],
            g=self._task_spec['g_dim'],
        )

        input_shapes = dims_to_shapes(self.input_dims)

        self.dimo = self._env_spec['o_dim']
        self.dimg = self._task_spec['g_dim']
        self.dima = self._env_spec['a_dim']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_next'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        self._action_noise, self._parameter_noise = get_noise_from_string(self._env_spec, noise_type)

        # Create network.
        with tf.variable_scope(self._scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        buffer_shapes = dict()
        buffer_shapes['o'] = (self.dimo,)
        buffer_shapes['o_next'] = buffer_shapes['o']
        buffer_shapes['g'] = (self.dimg,)
        buffer_shapes['ag'] = (self.dimg,)
        buffer_shapes['ag_next'] = (self.dimg,)
        buffer_shapes['a'] = (self.dima,)

        self.sample_transitions = make_sample_her_transitions(self.replay_strategy, self.replay_k, self._task.reward_done_success)

        self._buffer = ReplayBuffer(buffer_shapes, self.buffer_size, self._T, self.sample_transitions)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dima))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = simple_goal_subtract(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None

    def pre_rollout(self):
        if self._parameter_noise is not None:
            self.adapt_param_noise()
            self.sess.run(self.perturbe_op, feed_dict={self.param_noise_stddev: self._parameter_noise.current_stddev})

    def get_actions(self, o, ag, g, noise_eps=None, random_eps=None, use_target_net=False,
                    compute_Q=False, success_rate=None, mode=TRAIN):
        g = self._task.mg_fn(g)
        ag = self._task.mg_fn(ag)
        o, g = self._preprocess_og(o, ag, g)
        if mode == EVAL:
            policy = self.target if use_target_net else self.main
        else:
            if self._parameter_noise is not None:
                policy = self.perturbed
            else:
                policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.a_tf: np.zeros((o.size // self.dimo, self.dima), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        if mode == EVAL:
            u = np.clip(u, -self.max_u, self.max_u)
        else:
            if self._action_noise is not None:
                noise = self._action_noise()
                assert u.shape[0] == 1
                u = u[0]
                u += noise
                u = np.clip(u, -self.max_u, self.max_u)
            if self._parameter_noise is None and self._action_noise is None:
                if noise_eps is None: noise_eps = self.noise_eps
                if random_eps is None: random_eps = self.random_eps
                if self._noise_adaptation:
                    noise_eps *= 1 - success_rate
                    random_eps *= 1 - success_rate
                noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
                u += noise
                u = np.clip(u, -self.max_u, self.max_u)
                u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_transitions(self, episode, info, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        new_episode = episode
        if not self._share_experience:
            new_episode = {key: [np.asarray(value[i][np.squeeze(episode['tasks'][i]) == self._task_spec['id']])
                                 for i in range(len(value)) if np.any(np.squeeze(episode['tasks'][i]) == self._task_spec['id'])]
                         for key, value in episode.items()}


            batch_sizes = [len(value) for value in new_episode.values()]
            assert np.all(np.array(batch_sizes) == batch_sizes[0])
            batch_size = batch_sizes[0]
            if batch_size == 0: return

        new_batch = deepcopy(dict(
            o=new_episode['o'],
            o_next=new_episode['o_next'],
            a=new_episode['a'],
            ag=[self._task.mg_fn(ag) for ag in new_episode['ag']],
            ag_next=[self._task.mg_fn(ag_next) for ag_next in new_episode['ag_next']],
            g=[self._task.mg_fn(g) for g in new_episode['g']],
            g_next=[self._task.mg_fn(g_next) for g_next in new_episode['g_next']],
            r=new_episode['r'],
        ))

        self._buffer.store_episode(new_batch)

        if update_stats:
            num_normalizing_transitions = transitions_in_episode_batch(new_batch)
            new_batch['ep_T'] = np.asarray([ep.shape[0] for ep in list(new_batch.values())[0]])
            new_batch = {key: np.asarray(value) for key, value in new_batch.items()}
            transitions = self.sample_transitions(new_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self._buffer.get_current_size()

    def sample_batch(self):
        transitions = self._buffer.sample(self.batch_size) #otherwise only sample from primary buffer

        o, o_next, g = transitions['o'], transitions['o_next'], transitions['g']
        ag, ag_next = transitions['ag'], transitions['ag_next']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_next'], transitions['g_next'] = self._preprocess_og(o_next, ag_next, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        Q_loss, pi_loss, Q, q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.target.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        return Q_loss, pi_loss, Q, q_grad, pi_grad

    def _update(self, q_grad, pi_grad):
        self.Q_adam.update(q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def train(self, stage=True):
        if self._buffer.get_current_size() == 0: return {}
        Q_losses = []
        pi_losses = []
        Qs = []
        for i in range(self.train_steps):
            if stage:
                self.stage_batch()
            Q_loss, pi_loss, Q, q_grads, pi_grads = self._grads()
            self._update(q_grads, pi_grads)
            # Q_loss, pi_loss, Q, q_grads, pi_grads, *_ = self.sess.run([self.Q_loss_tf, self.main.Q_pi_tf,
            #                                                             self.target.Q_pi_tf,
            #                                                             self.Q_grads_tf, self.pi_grads_tf,
            #                                                             self.Q_train_op, self.pi_train_op])
            Q_losses.append(Q_loss)
            pi_losses.append(np.mean(pi_loss))
            Qs.append(Q)
        self.update_target_net()
        return {'Q_loss': ('scalar', np.mean(Q_losses)),
                'pi_loss': ('scalar', np.mean(pi_losses)),
                'Q': ('hist', np.hstack(Qs)),
                'q_grads': ('hist', np.hstack([q_grad.reshape(-1) for q_grad in q_grads])),
                'pi_grads': ('hist', np.hstack([pi_grad.reshape(-1) for pi_grad in pi_grads])),
                }

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self._buffer.clear_buffer()

    def _create_network(self, reuse=False):
        self.sess = tf.get_default_session() or tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        ))

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.normalizer_params['eps'], self.normalizer_params['default_clip_range'], sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.normalizer_params['eps'], self.normalizer_params['default_clip_range'], sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        network_params = deepcopy(self.network_params)
        del network_params['net_type']
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **network_params, **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_next']
            target_batch_tf['g'] = batch_tf['g_next']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **network_params, **self.__dict__)
            vs.reuse_variables()
        assert len(self.vars("main")) == len(self.vars("target"))
        if self._parameter_noise is not None:
            with tf.variable_scope('perturbed') as vs:
                if reuse:
                    vs.reuse_variables()
                self.perturbed = self.create_actor_critic(batch_tf, net_type='perturbed', **network_params, **self.__dict__)
                vs.reuse_variables()
            assert len(self.vars("main")) == len(self.vars("perturbed"))
            with tf.variable_scope('adaptive_param_noise') as vs:
                if reuse:
                    vs.reuse_variables()
                self.adaptive_param_noise = self.create_actor_critic(batch_tf, net_type='adaptive_param_noise', **network_params, **self.__dict__)
                vs.reuse_variables()
            assert len(self.vars("main")) == len(self.vars("adaptive_param_noise"))

            self.adaptive_param_noise_distance = tf.sqrt(tf.reduce_mean(tf.square(self.main.pi_tf - self.adaptive_param_noise.pi_tf)))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self._gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))

        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        self.Q_grads_tf = tf.gradients(self.Q_loss_tf, self.vars('main/Q'))
        self.pi_grads_tf = tf.gradients(self.pi_loss_tf, self.vars('main/pi'))
        assert len(self.vars('main/Q')) == len(self.Q_grads_tf)
        assert len(self.vars('main/pi')) == len(self.pi_grads_tf)
        self.Q_grads_vars_tf = zip(self.Q_grads_tf, self.vars('main/Q'))
        self.pi_grads_vars_tf = zip(self.pi_grads_tf, self.vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=self.Q_grads_tf, var_list=self.vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=self.pi_grads_tf, var_list=self.vars('main/pi'))

        self.Q_adam = MpiAdam(self.vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self.vars('main/pi'), scale_grad_by_procs=False)

        # self.Q_adam = tf.train.AdamOptimizer(learning_rate=self.Q_lr)
        # self.pi_adam = tf.train.AdamOptimizer(learning_rate=self.pi_lr)

        # self.Q_train_op = self.Q_adam.minimize(self.Q_loss_tf, var_list=self.vars('main/Q'))
        # self.pi_train_op = self.pi_adam.minimize(self.pi_loss_tf, var_list=self.vars('main/pi'))

        # polyak averaging
        self.main_vars = self.vars('main/Q') + self.vars('main/pi')
        self.target_vars = self.vars('target/Q') + self.vars('target/pi')
        self.stats_vars = self.global_vars('o_stats') + self.global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # perturbe
        if self._parameter_noise is not None:
            self.perturbed_vars = self.vars('perturbed/Q') + self.vars('perturbed/pi')
            self.perturbe_op = list(
                map(lambda v: v[0].assign(v[1] + tf.random_normal(tf.shape(v[1]), mean=0., stddev=self.param_noise_stddev)),
                    zip(self.perturbed_vars, self.main_vars))
            )

            self.adaptive_param_noise_vars = self.vars('adaptive_param_noise/Q') + self.vars('adaptive_param_noise/pi')
            self.adaptive_params_noise_perturbe_op = list(
                map(lambda v: v[0].assign(v[1] + tf.random_normal(tf.shape(v[1]), mean=0., stddev=self.param_noise_stddev)),
                    zip(self.adaptive_param_noise_vars, self.main_vars))
            )

        # initialize all variables
        tf.variables_initializer(self.global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

        self._sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.vars('')]
        self._sync_op_tf = [target.assign(source) for target, source in zip(self.vars(''), self._sync_source_tf)]

        self._global_sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.global_vars('')]
        self._global_sync_op_tf = [target.assign(source) for target, source in zip(self.global_vars(''), self._global_sync_source_tf)]

    def adapt_param_noise(self):

        if not self._buffer.get_current_size() > 0: return

        self.sess.run(self.adaptive_params_noise_perturbe_op, feed_dict={self.param_noise_stddev: self._parameter_noise.current_stddev})

        self.stage_batch()
        distance = self.sess.run(self.adaptive_param_noise_distance)

        self._parameter_noise.adapt(distance)

    def vars(self, scope=''):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)
        if scope == '':
            res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'o_stats/mean')
            res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'o_stats/std')
            res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'g_stats/mean')
            res += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + 'g_stats/std')
        assert len(res) > 0
        return res

    def global_vars(self, scope=''):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + scope)
        return res

    def get_params(self, scope=''):
        return (self._scope, [(str(var), self.sess.run(var)) for var in self.vars(scope)])

    def get_global_params(self, scope=''):
        return (self._scope, [(str(var), self.sess.run(var)) for var in self.global_vars(scope)])

    def set_params(self, params, scope=''):
        params = [param[1] for param in params]
        self.sess.run(self._sync_op_tf, feed_dict=dict([(ph, param) for ph, param in zip(self._sync_source_tf, params)]))

    def set_global_params(self, params, scope=''):
        params = [param[1] for param in params]
        self.sess.run(self._global_sync_op_tf, feed_dict=dict([(ph, param) for ph, param in zip(self._global_sync_source_tf, params)]))

if __name__ == '__main__':
    from copy import deepcopy
    from policies.const import TRAIN, EVAL

    from config import basic_configure

    tf.set_random_seed(1)
    np.random.seed(1)

    params = dict(
        experiment='/is/sg/sblaes/research/repositories/public/rec_center/experiments/CWYC/experiments/pickandplace_one_task/her_lowerBaseline.json',
        json_string='{"jobdir":"/local_data/sblaes/research/phd/repositories/public/rec_center/results/CWYC/pickandplace/one_task/lower_baseline"}'
    )

    basic_conf, modules_fn, \
    summary_writer = basic_configure(**params)

    rollout_worker_fn = modules_fn['rollout_worker_fn']
    params = deepcopy(modules_fn)
    del params['rollout_worker_fn']
    rollout_worker_fn[1]['discard_modules_buffer'] = False
    rollout_worker = rollout_worker_fn[0](0, env_spec=basic_conf['env_spec'], **params, **rollout_worker_fn[1])

    rollout_worker._env.seed(1)

    task = rollout_worker._tasks[0]
    policy = rollout_worker._policies[0]

    env_steps = 0
    for ep in range(2000):
        episode, info = rollout_worker.generate_rollout(render=False, mode=TRAIN)

        episodes = {key: [episode[key]] for key in episode.keys()}
        infos = {key: [info[key]] for key in info.keys()}

        env_steps += np.sum([o.shape[0] for o in episodes['o']])

        task.store_transitions(deepcopy(episodes), deepcopy(infos))
        res = task.train()
        if env_steps % 50 == 0:
            summary_writer.add_scalar('success_rate', res['success_rate'][1], env_steps)
            summary_writer.add_scalar('success_rate_deriv', res['success_rate_deriv'][1], env_steps)

        policy.store_transitions(deepcopy(episodes), deepcopy(infos))
        res = policy.train()
        if env_steps % 50 == 0:
            summary_writer.add_scalar('Q_loss', res['Q_loss'][1], env_steps)
            summary_writer.add_scalar('pi_loss', res['pi_loss'][1], env_steps)
            summary_writer.add_histogram('Qs', res['Q'][1], env_steps)
            summary_writer.add_histogram('Q_grads', res['q_grads'][1], env_steps)
            summary_writer.add_histogram('pi_grads', res['pi_grads'][1], env_steps)

        if env_steps % 50 == 0:
            success = 0.0
            for _ in range(10):
                episode, info = rollout_worker.generate_rollout(render=True, mode=EVAL)
                success += float(episodes['success'][0][-1])
            summary_writer.add_scalar('test/success_rate', success/10.0, env_steps)


