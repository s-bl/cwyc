import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from collections import OrderedDict
from copy import deepcopy

from normalizer import Normalizer
from parallel_module import ParallelModule
from simple_buffer import SimpleBuffer
from utils import to_transitions, import_function

class Gnet(ParallelModule):
    def __init__(self, env_spec, tasks_specs, task_from_id, task_to_id, pos_buffer_size=int(1e3), neg_buffer_size=int(1e5),
                 network_params=None, normalizer_params=None, coords_gen_params=None, batch_size=64, learning_rate=1e-4,
                 train_steps=1, only_fst_surprising_singal=True, reset_model_below_n_pos_samples=20,
                 only_pos_rollouts=False, normalize=False, use_switching_reward=True, scope='gnet'):
        super().__init__(scope)

        self._env_spec = env_spec
        self._task_to_id = task_to_id
        self._task_from_id = task_from_id
        self._tasks_specs = tasks_specs
        self._stage_shapes = OrderedDict(
            o=(self._env_spec['o_dim'],),
            r=(1,),
        )
        self._buffer = dict(
            pos_buffer=SimpleBuffer(self._stage_shapes, pos_buffer_size),
            neg_buffer=SimpleBuffer(self._stage_shapes, neg_buffer_size),
        )

        self._normalize = normalize
        self._normalizer_params = normalizer_params or dict()
        self._network_params = network_params or dict()
        self._coords_gen_params = coords_gen_params or dict()
        if pos_buffer_size == 0 and neg_buffer_size == 0:
            self._coords_gen_params['buffer_size'] = 0

        self._only_fst_surprising_singal = only_fst_surprising_singal
        self._only_pos_rollouts = only_pos_rollouts

        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._train_steps = train_steps
        self._reset_model_below_n_pos_samples = reset_model_below_n_pos_samples
        self._has_new_pos_samples = False
        self._use_switching_reward = use_switching_reward

        self._sess = tf.get_default_session() or tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        ))
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._staging_area_tf, self._input_ph_tf, self._staging_op_tf, self._batch_tf = self._prepare_staging()
            self._normalizer_tf = self._prepare_normalizers()
            self._net_tf = self._prepare_network()
            self._loss_tf, self._adam_tf, self._train_op_tf = self._prepare_training()

            self._sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.vars('')]
            self._sync_op_tf = [target.assign(source) for target, source in zip(self.vars(''), self._sync_source_tf)]

            self._global_sync_source_tf = [tf.placeholder(tf.float32, x.shape) for x in self.global_vars('')]
            self._global_sync_op_tf = [target.assign(source) for target, source in zip(self.global_vars(''), self._global_sync_source_tf)]

        self._sess.run(tf.variables_initializer(self.global_vars()))

    def _prepare_normalizers(self):
        normalizer_tf = dict()
        with tf.variable_scope('o_stats'):
            normalizer_tf['o'] = Normalizer(self._env_spec['o_dim'], **self._normalizer_params)

        return normalizer_tf

    def _prepare_staging(self):
        with tf.variable_scope('staging', reuse=tf.AUTO_REUSE):
            staging_area_tf = StagingArea(
                dtypes=[tf.float32 for _ in self._stage_shapes.keys()],
                shapes=[(None,*shape) for shape in self._stage_shapes.values()]
            )
            input_ph_tf = [tf.placeholder(tf.float32, shape=(None,*shape))
                           for shape in self._stage_shapes.values()]
            staging_op_tf = staging_area_tf.put(input_ph_tf)

            batch_tf = OrderedDict([(key, batch_item)
                                    for key, batch_item in zip(self._stage_shapes.keys(), staging_area_tf.get())])

        return staging_area_tf, input_ph_tf, staging_op_tf, batch_tf

    def _prepare_network(self):
        net_type = import_function(self._network_params['net_type'])
        self._logger.debug(f'Using {self._network_params["net_type"]}')
        params = deepcopy(self._network_params)
        del params['net_type']
        params['task_from_id'] = self._task_from_id
        params['task_to_id'] = self._task_to_id
        params['tasks_specs'] = self._tasks_specs
        net = net_type(self, self._batch_tf, self._normalizer_tf, **params)
        return net

    def _prepare_training(self):

        if len(self.vars()) == 0: return None, None, None

        adam_tf = tf.train.AdamOptimizer(learning_rate=self._learning_rate, epsilon=1e-4)
        loss_tf = tf.reduce_sum(tf.losses.get_losses(scope=self._scope))
        loss_tf += tf.losses.get_regularization_loss(scope=self._scope)
        train_op_tf = adam_tf.minimize(loss_tf, var_list=self.vars())

        return loss_tf, adam_tf, train_op_tf

    def _batch(self):
        pos_batch = self._buffer['pos_buffer'].sample(self._batch_size)
        neg_batch = self._buffer['neg_buffer'].sample(self._batch_size)

        batch = {key: np.vstack([pos_batch[key], neg_batch[key]]) for key in self._stage_shapes.keys()}
        return batch

    def _stage_batch(self):
        batch = self._batch()
        batch = [batch[key] for key in self._stage_shapes.keys()]
        self._sess.run(self._staging_op_tf, feed_dict=dict(zip(self._input_ph_tf, batch)))

    def train(self):
        if self._adam_tf is None: return {}

        out = dict(
            pos_buffer_size=('scalar', self._buffer['pos_buffer'].get_current_size()),
            neg_buffer_size=('scalar', self._buffer['neg_buffer'].get_current_size())
        )

        if self._buffer['pos_buffer'].get_current_size() == 0: return out
        if self._buffer['neg_buffer'].get_current_size() == 0: return out

        if self._has_new_pos_samples and self._buffer['pos_buffer'].get_current_size() < self._reset_model_below_n_pos_samples:
            self._sess.run(tf.variables_initializer(self.global_vars()))

        losses = []
        for i in range(self._train_steps):
            self._stage_batch()
            loss, _ = self._sess.run([self._loss_tf, self._train_op_tf])
            losses.append(loss)


        out.update({'loss': ('scalar', np.mean(losses))})

        self._net_tf.write_tflogs(scope=self._scope)

        self._has_new_pos_samples = False

        return out

    def _compute_switching_reward(self, batch, info):
        rs = []
        cut_off_idxs = []
        for ep in range(len(batch['o'])):
            r = np.zeros(batch['o'][ep].shape[0])
            switches = np.nonzero(batch['d'][ep])[0][:-1] # Last done signal is end of episode
                                                          # and not a switch

            for i, transition in enumerate(switches[:-1]):
                if info['task_seq'][ep][i] == self._task_from_id and info['task_seq'][ep][i+1] == self._task_to_id:
                    r[transition] = 1

            i = len(switches)-1
            if i >= 0:
                transition = switches[-1]
                if info['task_seq'][ep][i] == self._task_from_id and info['task_seq'][ep][i+1] == self._task_to_id:
                    # If we did not switched to final task, it can not be a success, otherwise
                    # check if we succeeded in final task
                    r[transition] = 0 if len(switches)+1 < len(info['task_seq'][ep]) else info['final_task_success'][ep]


            cut_off_idx = r.size
            if self._only_fst_surprising_singal:
                cut_off_idx = cut_off_idx if len(np.nonzero(r)[0]) == 0 else np.nonzero(r)[0][0]

            rs.append(r)
            cut_off_idxs.append(cut_off_idx)

        return rs, cut_off_idxs

    def _compute_surprise_reward(self, batch, info):
        rs = []
        cut_off_idxs = []
        for surprise, tasks in zip(batch['surprises'], batch['tasks']):
            # a surprising event happend in task task_to while doing task task_from
            r = np.asarray(surprise[:,self._task_to_id]) * (np.squeeze(tasks) == self._task_from_id).astype(np.int32)
            cut_off_idx = r.size
            if self._only_fst_surprising_singal:
                cut_off_idx = cut_off_idx if len(np.nonzero(r)[0]) == 0 else np.nonzero(r)[0][0]

            rs.append(r)
            cut_off_idxs.append(cut_off_idx)

        return rs, cut_off_idxs

    def store_transitions(self, batch, info):
        rs_surprise, cut_off_idxs_surprise = self._compute_surprise_reward(batch, info)
        rs_switching, cut_off_idxs_switching = self._compute_switching_reward(batch, info)

        if not self._use_switching_reward:
            rs_switching = [r*0 for r in rs_switching]
            cut_off_idxs_switching = [1000]*len(cut_off_idxs_switching)

        cut_off_idxs = [min(cut_off_idx_surprise, cut_off_idx_switching) for cut_off_idx_surprise, cut_off_idx_switching
                       in zip(cut_off_idxs_surprise, cut_off_idxs_switching)]
        rs = [np.minimum(r_surprise + r_switching, 1) for r_surprise, r_switching in zip(rs_surprise, rs_switching)]

        store_eps_idx = list(range(len(batch['o'])))
        if self._only_pos_rollouts:
            store_eps_idx = np.nonzero([(np.sum(r) > 0).astype(np.int32) for r in rs])[0]

        new_pos_batch = dict(
            o=[],
            r=[]
        )
        for i in range(len(batch['o'])):
            if i not in store_eps_idx: continue

            cut_off_idx = cut_off_idxs[i] + 1

            r = (rs[i][:cut_off_idx] > 0).astype(np.bool)
            o = batch['o'][i]
            o = o[:cut_off_idx]
            o = o[r]

            new_pos_batch['o'].append(o)
            new_pos_batch['r'].append(np.ones((o.shape[0],1)))

        if len(new_pos_batch['o']) > 0:
            new_pos_batch = {key: np.vstack(value) for key, value in new_pos_batch.items()}
            if new_pos_batch['o'].shape[0] > 0:
                self._buffer['pos_buffer'].store_transitions(new_pos_batch)
                self._update_normalizers(new_pos_batch)

        new_neg_batch = dict(
            o=[],
            r=[]
        )
        for i in range(len(batch['o'])):
            if i not in store_eps_idx: continue

            cut_off_idx = cut_off_idxs[i] + 1

            r = (rs[i][:cut_off_idx] == 0).astype(np.bool)
            o = batch['o'][i]
            o = o[:cut_off_idx]
            o = o[r]

            new_neg_batch['o'].append(o)
            new_neg_batch['r'].append(np.zeros((o.shape[0],1)))

        if len(new_neg_batch['o']) > 0:
            new_neg_batch = {key: np.vstack(value) for key, value in new_neg_batch.items()}
            if new_neg_batch['o'].shape[0] > 0:
                self._buffer['neg_buffer'].store_transitions(new_neg_batch)
                self._update_normalizers(new_neg_batch)

        if len(new_pos_batch['o']) > 0:
            self._has_new_pos_samples = True

    def _update_normalizers(self, batch):
        if not self._normalize: return
        transitions = to_transitions(batch)
        self._normalizer_tf['o'].update(transitions['o'])

        self._normalizer_tf['o'].recompute_stats()

    def _get_variable_coords(self, next_tasks_seq, prev_tasks_seq):
        variable_coords = set(self._tasks_specs[self._task_from_id]['mo'])

        mos = [task['mo'] for i, task in enumerate(self._tasks_specs)
               if i in prev_tasks_seq]

        if len(mos) > 0:
            variable_coords.update(np.hstack(mos))

        return list(variable_coords)

    def sample_goal(self, o, next_tasks_seq, prev_tasks_seq):
        variable_coords = self._get_variable_coords(next_tasks_seq, prev_tasks_seq)
        constant_coords = np.setdiff1d(np.arange(self._env_spec['o_dim']), variable_coords)

        goal =  self._net_tf.sample_goal(o, constant_coords=constant_coords, variable_coords=variable_coords)

        if 'goal_min' in self._env_spec:
            goal = np.clip(goal, self._env_spec['goal_min'][self._task_from_id], None)
        if 'goal_max' in self._env_spec:
            goal = np.clip(goal, None, self._env_spec['goal_max'][self._task_from_id])

        return goal


    def vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)

    def global_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + scope)

    def get_params(self, scope=''):
        return (self._scope, [(str(var), self._sess.run(var)) for var in self.vars(scope)])
        # return (self._scope, self._sess.run(self.vars(scope)) + [self._coords_gen.varCoords])

    def get_global_params(self, scope=''):
        return (self._scope, [(str(var), self._sess.run(var)) for var in self.global_vars(scope)])
        # return (self._scope, self._sess.run(self.global_vars()) + [self._coords_gen.varCoords])

    def set_params(self, params, scope=''):
        params = [param[1] for param in params]
        self._sess.run(self._sync_op_tf, feed_dict=dict([(ph, param) for ph, param in zip(self._sync_source_tf, params)]))
        # self._coords_gen.varCoords = params[-1]

    def set_global_params(self, params, scope=''):
        params = [param[1] for param in params]
        self._sess.run(self._global_sync_op_tf, feed_dict=dict([(ph, param) for ph, param in zip(self._global_sync_source_tf, params)]))
        # self._coords_gen.varCoords = params[-1]

if __name__ == '__main__':
    from config import basic_configure
    from functools import partial

    params = dict(
        experiment='/local_data/sblaes/research/phd/repositories/public/rec_center/experiments/CWYC/experiments/boxes/sac_json',
        json_string='{"jobdir":"/tmp/test"}'
    )

    basic_conf, modules_fn, summary_writer = basic_configure(**params)

    rollout_worker_fn = modules_fn['rollout_worker_fn']
    params = deepcopy(modules_fn)
    del params['rollout_worker_fn']
    rollout_worker_fn[1]['discard_modules_buffer'] = False
    rollout_worker = rollout_worker_fn[0](0, env_spec=basic_conf['env_spec'], **params, **rollout_worker_fn[1])

    prefix = 'latest'
    # path = '/local_data/sblaes/research/phd/repositories/public/rec_center/results/CWYC/boxes/cwyc/'
    path = '/is/sg/sblaes/remotes/is/cluster/experiments/rec_center/results/CWYC/boxes-sac_cwyc/0/'
    params = []
    for module in rollout_worker._tasks + rollout_worker._policies + [rollout_worker._gnets[i][j] for i in
                                                                      range(len(rollout_worker._gnets)) for j in
                                                                      range(len(rollout_worker._gnets))] + \
                  [rollout_worker._task_selector, rollout_worker._task_planner, rollout_worker._forward_model]:
        params.append(np.load(path + module._scope[:-len('_worker0')] + '_' + str(prefix) + '.npy'))
    params = dict(params)

    rollout_worker.set_global_params(params)


    def task_selector_sample_fun(final_task):
        def sample(self):
            return final_task

        rollout_worker._task_selector.sample = partial(sample, rollout_worker._task_selector)

    episodes, infos = [], []
    task_selector_sample_fun(2)
    for i in range(5):
        episode, info = rollout_worker.generate_rollout(render=False)
        episodes.append(episode)
        infos.append(info)

    episodes = {key: [episode[key] for episode in episodes] for key in episodes[0].keys()}
    infos = {key: [info[key] for info in infos] for key in infos[0].keys()}

    rollout_worker._gnets[0][1].store_transitions(episodes, infos)



