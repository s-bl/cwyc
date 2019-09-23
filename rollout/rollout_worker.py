import os
import logging
import numpy as np
import tensorflow as tf

from utils import get_jobdir
from policies.const import TRAIN, EVAL

import matplotlib.pyplot as plt # TODO: remove
from tensorboardX import SummaryWriter # TODO: remove

class RolloutWorker:
    def __init__(self, id, env_spec, env_fn, tasks_fn, policies_fn, gnets_fn, task_selector_fn, task_planner_fn,
                 forward_model_fn, seed, surprise_std_scaling=5, forward_model_burnin_eps=100, discard_modules_buffer=True,
                 early_stopping=True, resample_goal_every=5):

        self._id = id
        self._env_spec = env_spec
        self._T = self._env_spec['T']
        self._seed = seed
        self._discard_modules_buffer = discard_modules_buffer
        self._forward_model_burnin_eps = forward_model_burnin_eps
        self._early_stopping = early_stopping
        self._resample_goal_every = resample_goal_every

        self._env = env_fn[0](**env_fn[1])

        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)
        self._env.seed(self._id + self._seed)

        if self._discard_modules_buffer: [task_fn[1].update({'buffer_size':0}) for task_fn in tasks_fn]
        [task_fn[1].update({'scope': task_fn[1]['scope'] + f'_worker{id}'}) for task_fn in tasks_fn]
        self._tasks = [task_fn[0](**task_fn[1]) for task_fn in tasks_fn]

        if self._discard_modules_buffer: [policy_fn[1].update({'buffer_size':0}) for policy_fn in policies_fn]
        [policy_fn[1].update({'scope': policy_fn[1]['scope'] + f'_worker{id}'}) for policy_fn in policies_fn]
        self._policies = [policy_fn[0](**policy_fn[1]) for policy_fn in policies_fn]

        if self._discard_modules_buffer: [gnets_fn[i][j][1].update({'pos_buffer_size':0, 'neg_buffer_size':0}) for i in range(len(gnets_fn)) for j in range(len(gnets_fn))]
        [gnets_fn[i][j][1].update({'scope': gnets_fn[i][j][1]['scope'] + f'_worker{id}'}) for i in range(len(gnets_fn)) for j in range(len(gnets_fn))]
        self._gnets = [[gnets_fn[i][j][0](**gnets_fn[i][j][1]) for j in range(len(gnets_fn))] for i in range(len(gnets_fn))]

        if self._discard_modules_buffer: task_selector_fn[1]['buffer_size'] = 0
        task_selector_fn[1].update({'scope': task_selector_fn[1]['scope'] + f'_worker{id}'})
        self._task_selector = task_selector_fn[0](**task_selector_fn[1])

        if self._discard_modules_buffer: task_planner_fn[1]['buffer_size'] = 0
        task_planner_fn[1].update({'scope': task_planner_fn[1]['scope'] + f'_worker{id}'})
        self._task_planner = task_planner_fn[0](**task_planner_fn[1])

        if self._discard_modules_buffer: forward_model_fn[1]['buffer_size'] = 0
        forward_model_fn[1].update({'scope': forward_model_fn[1]['scope'] + f'_worker{id}'})
        self._forward_model = forward_model_fn[0](**forward_model_fn[1])

        self._surprise_std_scaling = surprise_std_scaling

        np.random.seed(id)

        self._logger = logging.getLogger(f'RolloutWorker{id:02}')

        self._jobdir = get_jobdir(self._logger)

        self._summary_writer = SummaryWriter(os.path.join(self._jobdir, f'tf_board/worker{self._id}'))

        self._ep = 0

    def set_params(self, params):
        for module in self._tasks + self._policies + [self._gnets[i][j] for i in range(len(self._gnets)) for j in range(len(self._gnets))] + \
            [self._task_selector, self._task_planner, self._forward_model]:
            scope = module._scope[:-len(f'_worker{self._id}')]
            if scope in params:
                module.set_params(params[scope])

    def set_global_params(self, params):
        for module in self._tasks + self._policies + [self._gnets[i][j] for i in range(len(self._gnets)) for j in range(len(self._gnets))] + \
                      [self._task_selector, self._task_planner, self._forward_model]:
            scope = module._scope[:-len(f'_worker{self._id}')]
            if scope in params:
                module.set_global_params(params[scope])

    def _reset(self, final_task=None):
        obs = self._env.reset()
        initial_goal = obs['desired_goal']

        if final_task is None:
            final_task = self._task_selector.sample()
        tasks_seq = [self._tasks[id] for id in self._task_planner.sample(final_task)]

        task_from = tasks_seq[0]
        task_to = tasks_seq[1] if len(tasks_seq) > 1 else None

        goal = self._set_goal(obs['observation'], initial_goal, task_from, task_to, tasks_seq[1:], [])
        obs['desired_goal'] = goal.copy()

        for policy in self._policies:
            if callable(getattr(policy, 'pre_rollout', None)):
                policy.pre_rollout()

        return obs, initial_goal, tasks_seq

    def _set_goal(self, o, initial_goal, task_from, task_to, next_tasks_seq, prev_tasks_seq):
        goal = initial_goal.copy()
        new_task_goal = task_from.mg_fn(goal)
        next_tasks_seq = [task.id for task in next_tasks_seq]
        prev_tasks_seq = [task.id for task in prev_tasks_seq] # TODO: Also considere past tasks
        if task_to is not None: # Get subgoal from gnet
            new_task_goal = self._gnets[task_from.id][task_to.id].sample_goal(o, next_tasks_seq, prev_tasks_seq)
            goal = task_from.assign_goal(goal, new_task_goal)
            # fig, ax = plt.subplots(1,1)
            # ax.scatter(new_task_goal[0], new_task_goal[1], marker='x')
            # if task_from.id == 0 and task_to.id == 1:
            #     ax.scatter(o[2], o[3], marker='o')
            # if task_from.id == 1 and task_to.id == 2:
            #     ax.scatter(o[4], o[5], marker='o')
            # ax.set_xlim([-15, 15])
            # ax.set_ylim([-15, 15])
            # self._summary_writer.add_figure(f'gnet_{task_from.id}_to_{task_to.id}/goal', fig, self._ep)
            # plt.close(fig)

        self._env.set_goal(goal, [task._mg for task in self._tasks])
        return goal

    def _get_action(self, o, ag, g, current_task_id, mode):
        return self._policies[current_task_id].get_actions(o[None,:], ag[None,:], g[None,:],
                                                           success_rate=self._tasks[current_task_id].success_rate,
                                                           mode=mode)

    def _compute_suprise(self, obs, actions, obs_next, tasks):
        pred = self._forward_model.predict(obs[:,None,:], actions[:,None,:], obs_next)
        abs_pred_error = np.abs(pred['prediction_error'])
        pred_err_mean = pred['abs_prediction_error_mean']
        pred_err_std = pred['abs_prediction_error_std']
        surprise_obs = (abs_pred_error >= pred_err_mean + self._surprise_std_scaling * pred_err_std).astype(np.int32)
        surprise_tasks = []
        for task in self._tasks:
            mo = self._tasks[task.id]._mo
            surprise_tasks.append(np.max(surprise_obs[:,mo], axis=1, keepdims=True) *
                                  self._ep >= self._forward_model_burnin_eps-1)
        surprise_tasks = np.hstack(surprise_tasks)

        for task_idx in range(len(self._tasks)):
            if np.max(surprise_tasks[:,task_idx]) > 0:
            #  if np.max(surprise_tasks[:,task_idx]) > 0 or (self._ep % 10 == 0 and self._ep >= self._forward_model_burnin_eps-1):
                colors = ['blue', 'orange', 'green']
                mo = self._tasks[task_idx]._mo
                fig, axs = plt.subplots(1,3,figsize=(12,5))
                axs[0].plot(obs_next[:,mo])
                axs[0].plot(pred['predicted_state'][:,mo])
                axs[0].plot(surprise_tasks[:,task_idx])
                [axs[1].plot(abs_pred_error[:,k], color=colors[i]) for i, k in enumerate(mo)]
                [axs[1].axhline(pred_err_mean[k] + self._surprise_std_scaling * pred_err_std[k], color=colors[i]) for i, k in enumerate(mo)]
                axs[2].plot(tasks)
                self._summary_writer.add_figure(f'worker{self._id}/pred_task{task_idx}', fig, self._ep)
                plt.close(fig)

        return surprise_tasks, pred

    def generate_rollout(self, final_task=None, render=False, mode=TRAIN):

        obs, initial_goal, tasks_seq = self._reset(final_task)

        self._logger.info(f'Solving task{tasks_seq[-1].id} via {[task.id for task in tasks_seq[:-1]]}')

        current_task_idx = 0
        current_task = tasks_seq[0]
        next_task = tasks_seq[1] if len(tasks_seq) > 1 else None

        observations = np.zeros((self._T, self._env_spec['o_dim']))
        observations_next = np.zeros((self._T, self._env_spec['o_dim']))
        actions = np.zeros((self._T, self._env_spec['a_dim']))
        achieved_goals = np.zeros((self._T, self._env_spec['g_dim']))
        achieved_goals_next = np.zeros((self._T, self._env_spec['g_dim']))
        desired_goals = np.zeros((self._T, self._env_spec['g_dim']))
        desired_goals_next = np.zeros((self._T, self._env_spec['g_dim']))
        rewards = np.zeros((self._T, 1))
        dones = np.zeros((self._T, 1))
        tasks = np.zeros((self._T, 1))
        success = np.zeros((self._T, 1))
        final_task_success = 0
        subtask_T = []

        done_with_final_task = False # used in case of no early stopping

        for t in range(self._T):
            a = self._get_action(obs['observation'], obs['achieved_goal'], obs['desired_goal'], current_task.id, mode)

            new_obs, reward, done, _ = self._env.step(a)

            observations[t] = obs['observation'].copy()
            observations_next[t] = new_obs['observation'].copy()
            actions[t] = a.copy()
            achieved_goals[t] = obs['achieved_goal'].copy()
            achieved_goals_next[t] = new_obs['achieved_goal'].copy()
            desired_goals[t] = obs['desired_goal'].copy()
            desired_goals_next[t] = new_obs['desired_goal'].copy()
            rewards[t], dones[t], success[t] = current_task.reward_done_success(ag=obs['achieved_goal'],
                                                                                a=a, new_ag=new_obs['achieved_goal'],
                                                                                new_g=new_obs['desired_goal'],  t=t,
                                                                                T=self._T
                                                                                )

            tasks[t] = current_task.id

            if self._resample_goal_every > 0 and t % self._resample_goal_every == 0:
                goal = self._set_goal(new_obs['observation'], initial_goal, current_task, next_task,
                                      tasks_seq[current_task_idx + 1:] if current_task_idx + 1 < len(
                                          tasks_seq) else [], tasks_seq[:current_task_idx])
                new_obs['desired_goal'] = goal.copy()
                desired_goals_next[t] = goal.copy()

            if dones[t] == 1:
                if next_task is None and self._early_stopping: # We are done with final task
                    final_task_success = int(success[t])
                    if final_task_success:
                        subtask_T.append(t+1)
                        self._logger.info(f'Solved task{tasks_seq[-1].id} in step {t}')
                        done_with_final_task = True
                    break
                elif next_task is None and not self._early_stopping:
                    dones[t] = 0
                    if int(success[t]) and not done_with_final_task:
                        subtask_T.append(t+1)
                        done_with_final_task = True
                elif int(success[t]): # We are done with subtask
                    self._logger.info(f'Completed subtask '
                                      f'[{", ".join([str(task.id) for task in tasks_seq[:current_task_idx]])}'
                                      f'{", " if current_task_idx > 0 else ""}'
                                      f'({tasks_seq[current_task_idx].id})'
                                      f'{", " if next_task is not None else ""}'
                                      f'{", ".join([str(task.id) for task in tasks_seq[current_task_idx+1:]])}] in step {t}')
                    current_task_idx +=1
                    current_task = tasks_seq[current_task_idx]
                    next_task = tasks_seq[current_task_idx+1] if current_task_idx+1 < len(tasks_seq) else None

                    goal = self._set_goal(new_obs['observation'], initial_goal, current_task, next_task,
                                          tasks_seq[current_task_idx + 1:] if current_task_idx + 1 < len(
                                              tasks_seq) else [], tasks_seq[:current_task_idx])
                    new_obs['desired_goal'] = goal.copy()
                    desired_goals_next[t] = goal.copy()
                    subtask_T.append(t+1)
                else:
                    pass

            obs = new_obs

            if render and self._id == 0:
                self._env.render()

        if not done_with_final_task:
            subtask_T.append(t+1)

        if not self._early_stopping:
            final_task_success = int(success[t]) * int(next_task is None)
            if done_with_final_task:
                self._logger.info(f'Solved task{tasks_seq[-1].id} in step {subtask_T[-1]}')
                dones[-1] = 1

        surprises, pred = self._compute_suprise(observations[:t + 1], actions[:t + 1], observations_next[:t + 1], tasks)

        episode = dict(
            o=observations[:t+1],
            o_next=observations_next[:t+1],
            a=actions[:t+1],
            ag=achieved_goals[:t+1],
            ag_next=achieved_goals_next[:t+1],
            g=desired_goals[:t+1],
            g_next=desired_goals_next[:t+1],
            r=rewards[:t+1],
            d=dones[:t+1],
            success=success[:t+1],
            tasks=tasks[:t+1],
            surprises = surprises[:t + 1],
        )
        info = dict(
            task_seq=[task.id for task in tasks_seq],
            final_task_success=final_task_success,
            success_rates=[task.success_rate for task in self._tasks],
            abs_success_rates_deriv=[task.abs_success_rate_deriv for task in self._tasks],
            subtask_T=subtask_T,
        )
        info.update(pred)

        self._ep += 1

        return episode, info

if __name__ == '__main__':
    from config import basic_configure
    from copy import deepcopy
    from functools import partial

    params = dict(
        experiment='/is/sg/sblaes/research/repositories/public/rec_center/experiments/CWYC/experiments/boxes/sac_json',
        json_string='{"jobdir":"/tmp/notebook1"}'
    )

    basic_conf, modules_fn, \
    summary_writer = basic_configure(**params)

    rollout_worker_fn = modules_fn['rollout_worker_fn']
    params = deepcopy(modules_fn)
    del params['rollout_worker_fn']
    rollout_worker_fn[1]['discard_modules_buffer'] = False
    rollout_worker = rollout_worker_fn[0](0, env_spec=basic_conf['env_spec'], **params, **rollout_worker_fn[1])

    # prefix = '950'
    # params = []
    # path = '/is/sg/sblaes/research/repositories/public/rec_center/results/CWYC/boxes/one_task/cwyc/'
    # for module in rollout_worker._tasks + rollout_worker._policies + [rollout_worker._gnets[i][j] for i in
    #                                                                   range(len(rollout_worker._gnets)) for j in
    #                                                                   range(len(rollout_worker._gnets))] + \
    #               [rollout_worker._task_selector, rollout_worker._task_planner, rollout_worker._forward_model]:
    #     params.append(np.load(os.path.join(path, module._scope[:-len('_worker0')] + '_' + str(prefix) + '.npy')))
    # params = dict(params)
    #
    # rollout_worker.set_params(params)


    def task_selector_sample_fun(final_task):
        def sample(self):
            return final_task

        rollout_worker._task_selector.sample = partial(sample, rollout_worker._task_selector)


    def task_planner_sample_fun(seq):
        def sample(self, final_task):
            return seq

        rollout_worker._task_planner.sample = partial(sample, rollout_worker._task_planner)

    task_selector_sample_fun(1)
    task_planner_sample_fun([1])
    while True:
        episode, info = rollout_worker.generate_rollout(render=True, mode=1)
