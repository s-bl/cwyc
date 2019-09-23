import numpy as np
from collections import deque
import os

from parallel_module import ParallelModule
from forward_model.utils import weighted_surprise
from utils import normalize

class TaskPlanner(ParallelModule):
    def __init__(self, env_specs, tasks_specs, surprise_weighting=0.1, surprise_hist_weighting=0.99, buffer_size=100,
                 eps_greedy_prob=0.05, max_seq_length=10, epsilon=0.0001, fixed_Q=None, scope='taskPlanner'):
        super().__init__(scope=scope)
        self._env_specs = env_specs
        self._T = self._env_specs['T']
        self._tasks_specs = tasks_specs
        self._num_tasks = len(self._tasks_specs)
        self._surprise_weighting = surprise_weighting
        self._surprise_hist_weighting = surprise_hist_weighting
        self._eps_greedy_prob = eps_greedy_prob
        self._max_seq_lenght = max_seq_length
        self._epsilon = epsilon
        self._buffer = dict(
            transitions_time=[[deque(maxlen=buffer_size) for _ in range(self._num_tasks)]
                               for _ in range(self._num_tasks)],
            surprises=[[deque(maxlen=buffer_size) for _ in range(self._num_tasks)]
                               for _ in range(self._num_tasks)],
        )
        _ = [[self._buffer['transitions_time'][i][j].append(self._T) for j in range(self._num_tasks)]
             for i in range(self._num_tasks)]
        self._Q = np.zeros((self._num_tasks, self._num_tasks))
        self._fixed_Q = np.asarray(fixed_Q).T if fixed_Q is not None else None

    def store_transitions(self, batch, info):
        if self._fixed_Q is None:
            for subtask_T, task_seq in zip(info['subtask_T'], info['task_seq']):
                if len(task_seq) == 1:
                    self._buffer['transitions_time'][task_seq[0]][task_seq[0]].append(subtask_T[0])
                for i, subtask_t in enumerate(subtask_T[1:]):
                    self._buffer['transitions_time'][task_seq[i]][task_seq[i + 1]].append(subtask_t)
                    #
                    # if task_seq[i] in [1,2] and task_seq[i+1] == 0 and subtask_t < self._T:
                    #     self._logger.warning(f'{subtask_T},{task_seq}')

            for ep in range(len(batch['surprises'])):
                for task_from in range(self._num_tasks):
                    for task_to in range(self._num_tasks):
                        self._buffer['surprises'][task_from][task_to].append(np.max(batch['surprises'][ep][:,task_to][:,None] *
                                                                             (batch['tasks'][ep] == task_from).astype(np.int32)))

    def train(self):
        if self._fixed_Q is None:
            batch = self._buffer
            mean_transition_times = np.asarray([[np.mean(batch['transitions_time'][i][j]) for j in range(self._num_tasks)]
                                     for i in range(self._num_tasks)])

            Q_t = 1 - (mean_transition_times / self._T)

            # Find last surprise signal and weight according to how long it is in the past
            surprises_flat = np.asarray([batch['surprises'][i][j] for i in range(self._num_tasks)
                                     for j in range(self._num_tasks)]).T
            batch_len = surprises_flat.shape[0]

            w_surprise = weighted_surprise(surprises_flat, self._surprise_hist_weighting, batch_len).reshape(self._num_tasks, self._num_tasks)
            Q_surprise = self._surprise_weighting * w_surprise
            self._Q = Q_t + Q_surprise + self._epsilon


        out = dict()
        if self._fixed_Q is not None:
            Q = self._fixed_Q
        else:
            Q = self._Q

            out.update({f'task_{i}_to_{j}/mean_transition_time': ('scalar', mean_transition_times[i,j]) for i in range(self._num_tasks) for j in range(self._num_tasks)})
            out.update({f'task_{i}_to_{j}/surprise': ('scalar', Q_surprise[i,j]) for i in range(self._num_tasks) for j in range(self._num_tasks)})

        out.update({f'task_{i}_to_{j}/Q': ('scalar', Q[i][j]) for i in range(self._num_tasks) for j in range(self._num_tasks)})
        out.update({f'task_{i}_to_{j}/prob': ('scalar', p) for j in range(self._num_tasks) for i, p in enumerate(normalize(Q[:,j]))})

        return out

    def sample(self, final_task):
        if self._fixed_Q is not None:
            Q = self._fixed_Q
        else:
            Q = self._Q

        previous_task = final_task
        next_task = None
        task_seq = [final_task]
        while len(task_seq) <= self._max_seq_lenght:
            if np.random.rand() < self._eps_greedy_prob and self._fixed_Q is None: # act randomly
                next_task = np.random.choice(range(self._num_tasks), p=normalize(Q[:, previous_task]))
            else: # act greedily
                next_task = np.argmax(normalize(Q[:, previous_task]))

            if next_task == previous_task: break
            if len(task_seq) > 1 and (next_task, task_seq[0]) in [(x, y) for x, y in zip(task_seq[:-1], task_seq[1:])]:
                break # TODO: Same task transition should not appear twice
            # if next_task in task_seq: break # TODO: Task should not appear twice in sequence

            task_seq = [next_task] + task_seq
            previous_task = next_task

        return task_seq

    def get_global_params(self, scope=''):
        return (self._scope, [('Q', self._Q)])

    def get_params(self, scope=''):
        return self.get_global_params(scope)

    def set_global_params(self, params, scope=''):
        params = [param[1] for param in params]
        self._Q = params[0]

    def set_params(self, params, scope=''):
        self.set_global_params(params, scope)

    def save_buffer(self, path, prefix=''):
        prefix = '' if not prefix else f'_{prefix}'
        np.save(os.path.join(path, f'{self._scope}_buffer{prefix}'), [self._buffer['transitions_time'], self._buffer['surprises']])

    def restore_buffer(self, path, prefix=''):
        prefix = '' if not prefix else f'_{prefix}'
        transition_time, surprises = np.load(os.path.join(path, f'{self._scope}_buffer{prefix}'))
        self._buffer['transitions_time'].extend(transition_time)
        self._buffer['surprises'].extend(surprises)

if __name__ == '__main__':
    import dill
    from config import basic_configure
    from copy import deepcopy
    from functools import partial

    params = dict(
        experiment='/local_data/sblaes/research/phd/repositories/public/rec_center/experiments/CWYC/experiments/pickandplace/three_task_her_json',
        json_string='{"jobdir":"/tmp/test"}'
    )

    basic_conf, modules_fn, summary_writer = basic_configure(**params)

    rollout_worker_fn = modules_fn['rollout_worker_fn']
    params = deepcopy(modules_fn)
    del params['rollout_worker_fn']
    rollout_worker = rollout_worker_fn[0](0, env_spec=basic_conf['env_spec'], **params, **rollout_worker_fn[1])

    prefix = 'latest'
    params = []
    for module in rollout_worker._tasks + rollout_worker._policies + [rollout_worker._gnets[i][j] for i in
                                                                      range(len(rollout_worker._gnets)) for j in
                                                                      range(len(rollout_worker._gnets))] + \
                  [rollout_worker._task_selector, rollout_worker._task_planner, rollout_worker._forward_model]:
        params.append(np.load(
            '/local_data/sblaes/research/phd/repositories/public/rec_center/results/CWYC/pickandplace/three_tasks/cwyc/' + module._scope + '_' + str(
                prefix) + '.npy'))
    params = dict(params)


    def task_selector_sample_fun(final_task):
        def sample(self):
            return final_task

        rollout_worker._task_selector.sample = partial(sample, rollout_worker._task_selector)


    task_selector_sample_fun(0)

    episode, info = rollout_worker.generate_rollout(render=False)
