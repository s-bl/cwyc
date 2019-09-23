import logging
from multiprocessing import Pipe, Process
from policies.const import TRAIN, EVAL


class ParallelRolloutManager:
    def __init__(self, env_spec, env_fn, tasks_fn, policies_fn, gnets_fn, task_selector_fn, task_planner_fn,
                 forward_model_fn, rollout_worker_fn, T, num_worker,
                 managed_memory):
        super().__init__()
        logger = logging.getLogger('ParallelRolloutWorker')
        self._env_spec = env_spec
        self._env_fn = env_fn
        self._num_workers = num_worker
        self._tasks_fn = tasks_fn
        self._num_tasks = len(self._tasks_fn)
        self._policies_fn = policies_fn
        self._gnets_fn = gnets_fn
        self._task_selector_fn = task_selector_fn
        self._task_planner_fn = task_planner_fn
        self._forward_model_fn = forward_model_fn
        self._rollout_worker_fn = rollout_worker_fn
        self._managed_memory = managed_memory

        self._T = T

        self._remotes, self._work_remotes = zip(*[Pipe() for _ in range(self._num_workers)])
        self._ps = [Process(target=self.run, args=(worker, work_remote, remote, self._env_spec, self._env_fn,
                                                   self._tasks_fn, self._policies_fn,
                                                   self._gnets_fn, self._task_selector_fn, self._task_planner_fn,
                                                   self._forward_model_fn, self._rollout_worker_fn))
                   for (worker, work_remote, remote) in zip(range(self._num_workers), self._work_remotes, self._remotes)]

        self._waiting = False
        self._closed = False
        for p in self._ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self._work_remotes:
            remote.close()

    def run(self, worker, remote, parent_remote, env_spec, env_fn, tasks_fn, policies_fn, gnets_fn, task_selector_fn, task_planner_fn,
            forward_model_fn, rollout_worker_fn):
        parent_remote.close()
        worker = rollout_worker_fn[0](worker, env_spec, env_fn, tasks_fn, policies_fn, gnets_fn, task_selector_fn, task_planner_fn,
                               forward_model_fn, **rollout_worker_fn[1])
        while True:
            cmd, data = remote.recv()
            if cmd == 'generate_rollout':
                render = data[0]
                final_task = data[1]
                mode = data[2]
                episode, info = worker.generate_rollout(final_task, render, mode)
                remote.send((episode, info))
            elif cmd == 'set_params':
                params = data[0]
                worker.set_params(params)
                remote.send(True)
            elif cmd == 'set_global_params':
                params = data[0]
                worker.set_global_params(params)
                remote.send(True)
            elif cmd == 'set_ep':
                ep = data[0]
                worker._ep = ep
                remote.send(True)
            elif cmd == 'quit':
                remote.close()
                break

    def generate_rollouts(self, final_tasks=None, render=False, mode=TRAIN):
        for i, remote in enumerate(self._remotes):
            final_task = None
            if isinstance(final_tasks, int):
                final_task = final_tasks
            if isinstance(final_tasks, list):
                assert len(final_tasks) == len(self._remotes)
                final_task = final_tasks[i]
            remote.send(('generate_rollout', (render,final_task,mode)))
        episode_res, info_res = zip(*[remote.recv() for remote in self._remotes])
        for key in episode_res[0].keys():
            self._managed_memory['episode'][key] = [episode_res[i][key] for i in range(self._num_workers)]
        for key in info_res[0].keys():
            self._managed_memory['info'][key] = [info_res[i][key] for i in range(self._num_workers)]

    def set_params(self, params):
        for remote in self._remotes:
            remote.send(('set_params', (params,)))
        res = [remote.recv() for remote in self._remotes]

    def set_global_params(self, params):
        for remote in self._remotes:
            remote.send(('set_global_params', (params,)))
        res = [remote.recv() for remote in self._remotes]

    def set_ep(self, ep):
        for remote in self._remotes:
            remote.send(('set_ep', (ep,)))
        res = [remote.recv() for remote in self._remotes]

    def __del__(self):
        for remote in self._remotes:
            remote.send(('quit', None))

if __name__ == '__main__':
    from task.task import Task
    from envs.continuous.BallPickUpExtendedObstacles import Ball
    from task_selector.task_selector import TaskSelector
    from task_planner.task_planner import TaskPlanner
    from policies.sac.sac import SAC
    from multiprocessing import Manager
    import dill

    logging.basicConfig(level=logging.INFO)

    class Gnet:
        def __init__(self, task_from, task_to, scope='gnet'):
            self._scope = scope
            self._task_from = task_from
            self._task_to = task_to

        def sample_goal(self, o):
            logger.info(f'Going from task {self._task_from.id} to task {self._task_to.id}. mg: {self._task_to._mg}')
            return self._task_to.mg_fn(o)

    tmp_env = Ball()
    obs = tmp_env.reset()
    env_spec = dict(
        o_dim=obs['observation'].shape[0],
        a_dim=tmp_env.action_space.shape[0],
        g_dim=obs['desired_goal'].shape[0],
        og_dim=obs['observation'].shape[0]+obs['desired_goal'].shape[0],
    )
    tasks_fn = []
    tasks_fn.append((Task, dict(id=0, mg=range(0,2), th=0.5)))
    tasks_fn.append((Task, dict(id=1, mg=range(2,4), th=0.5)))
    env_fn = Ball
    policies_fn = [(SAC, dict(env_spec=env_spec)), (SAC, dict(env_spec=env_spec))]
    task_selector_fn = TaskSelector
    task_planner_fn = TaskPlanner
    gnets_fn = [[(Gnet, dict()) for i in range(2)] for j in range(2)]
    T = 300
    num_worker = 5

    manager = Manager()
    episode = manager.dict()
    info = manager.dict()
    managed_memory = dict(
        episode=episode,
        info=info,
    )


    rollout_worker = ParallelRolloutManager(env_spec, env_fn, tasks_fn, policies_fn, gnets_fn, task_selector_fn,
                                           task_planner_fn, forward_model_fn, T, num_worker, managed_memory)
    env, tasks, policies, gnets, task_selector, task_planner = init_workers(0, env_fn, tasks_fn, policies_fn,
                                                                            gnets_fn, task_selector_fn,
                                                                            task_planner_fn)
    episode, info = rollout_worker._generate_rollout(0, env, tasks, policies, gnets, task_selector, task_planner, False)

    with open('/tmp/episode.pkl', 'wb') as f:
        dill.dump([episode, info], f)

    # rollout_worker.init_workers()
    # rollout_worker.generate_rollouts(False)
    # print('Done')

