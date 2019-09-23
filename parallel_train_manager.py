from multiprocessing import Process, Pipe
import logging
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

class ParallelTrainManager:
    def __init__(self, managed_memory, summary_writer, jobdir, seed):

        self._managed_memory = managed_memory
        self._jobdir = jobdir
        self._seed = seed

        self._summary_writer = summary_writer

        self._remotes = []
        self._ps = []

    def add_module(self, module_fn):
        remote, work_remote = Pipe()
        p = Process(target=self.run, args=(work_remote, remote, module_fn, self._managed_memory, self._seed))

        p.deamon = True
        p.start()
        work_remote.close()

        self._remotes.append(remote)
        self._ps.append(p)

        self._train_res = None

        return len(self._ps)-1

    def remove_module(self, module_idx):
        self._remotes[module_idx].send(('close', None))
        self._ps[module_idx].join()

        self._remotes.pop(module_idx)
        self._ps.pop(module_idx)

    def run(self, remote, parent_remote, module_fn, managed_memory, seed):
        parent_remote.close()
        module = module_fn[0](**module_fn[1])
        module.set_seed(seed)
        while True:
            cmd, data = remote.recv()
            if cmd == 'train':
                res = module.train()
                remote.send((module._scope, res))
            elif cmd == 'get_params':
                params = module.get_params('')
                remote.send(params)
            elif cmd == 'get_global_params':
                params = module.get_global_params('')
                remote.send(params)
            elif cmd == 'store_transitions':
                module.store_transitions(deepcopy(managed_memory['episode']), deepcopy(managed_memory['info']))
                remote.send(True)
            elif cmd == 'save_params':
                path = data[0]
                prefix = data[1]
                res = module.save_params(path, prefix)
                remote.send(res)
            elif cmd == 'save_global_params':
                path = data[0]
                prefix = data[1]
                res = module.save_global_params(path, prefix)
                remote.send(res)
            elif cmd == 'load_params':
                path = data[0]
                prefix = data[1]
                module.load_params(path, prefix)
                remote.send(True)
            elif cmd == 'load_global_params':
                path = data[0]
                prefix = data[1]
                module.load_global_params(path, prefix)
                remote.send(True)
            elif cmd == 'save_buffer':
                path = data[0]
                prefix = data[1]
                module.save_buffer(path, prefix)
                remote.send(True)
            elif cmd == 'restore_buffer':
                path = data[0]
                prefix = data[1]
                module.restore_buffer(path, prefix)
                remote.send(True)
            else:
                remote.close()
                break

    def init(self):
        for remote in self._remotes:
            remote.send(('init', None))
        res = [remote.recv() for remote in self._remotes]
        logger.debug('Initialized networks')

    def train(self):
        for remote in self._remotes:
            remote.send(('train', None))
        res = [remote.recv() for remote in self._remotes]
        self._train_res = dict(res)

    def get_params(self):
        for remote in self._remotes:
            remote.send(('get_params', None))
        params = [remote.recv() for remote in self._remotes]
        return dict(params)

    def get_global_params(self):
        for remote in self._remotes:
            remote.send(('get_global_params', None))
        params = [remote.recv() for remote in self._remotes]
        return dict(params)

    def store_transitions(self):
        for remote in self._remotes:
            remote.send(('store_transitions', None))
        res = [remote.recv() for remote in self._remotes]

    def save_params(self, prefix=''):
        for remote in self._remotes:
            remote.send(('save_params', (self._jobdir, prefix)))
        res = [remote.recv() for remote in self._remotes]
        return res

    def save_global_params(self, prefix=''):
        for remote in self._remotes:
            remote.send(('save_global_params', (self._jobdir, prefix)))
        res = [remote.recv() for remote in self._remotes]
        return res

    def load_params(self, path=None, prefix=''):
        jobdir = path or self._jobdir
        for remote in self._remotes:
            remote.send(('load_params', (jobdir, prefix)))
        res = [remote.recv() for remote in self._remotes]

    def load_global_params(self, path=None, prefix=''):
        jobdir = path or self._jobdir
        for remote in self._remotes:
            remote.send(('load_global_params', (jobdir, prefix)))
        res = [remote.recv() for remote in self._remotes]

    def save_buffer(self, prefix=''):
        for remote in self._remotes:
            remote.send(('save_buffer', (self._jobdir, prefix)))
        res = [remote.recv() for remote in self._remotes]

    def restore_buffer(self, path=None, prefix=''):
        jobdir = path or self._jobdir
        for remote in self._remotes:
            remote.send(('restore_buffer', (jobdir, prefix)))
        res = [remote.recv() for remote in self._remotes]

    def _write_data(self, tag, dtype, data, ep, env_step):
        if dtype == 'scalar':
            self._summary_writer.add_scalar(tag, data, env_step)
        elif dtype == 'hist':
            self._summary_writer.add_histogram(tag, data, env_step)

    def write_tflogs(self, keys=None, ep=0, env_step=0):
        assert self._train_res is not None
        if keys is None:
            for module in self._train_res.keys():
                if self._train_res[module] is not None:
                    for val in self._train_res[module].keys():
                        self._write_data(module+'/'+val, self._train_res[module][val][0], self._train_res[module][val][1], ep, env_step)
        else:
            for key in keys:
                module, val = key.split('/')
                assert module in self._train_res and val in self._train_res[module]
                self._write_data(key, self._train_res[module][val][0], self._train_res[module][val][1], ep, env_step)

    def __del__(self):
        for remote in self._remotes:
            remote.send(('quit', None))


if __name__ == '__main__':
    from config import basic_configure
    from multiprocessing import Manager

    basic_conf = basic_configure(jobdir='/tmp/test', clean=True)

    env_spec = basic_conf['env_spec']
    T = env_spec['T']
    env_fn = basic_conf['env_fn']
    tasks_fn = basic_conf['tasks_fn']
    policies_fn = basic_conf['policies_fn']
    gnets_fn = basic_conf['gnets_fn']
    task_selector_fn = basic_conf['task_selector_fn']
    task_planner_fn = basic_conf['task_planner_fn']
    forward_model_fn = basic_conf['forward_model_fn']
    rollout_worker_fn = basic_conf['rollout_worker_fn']
    summary_writer = basic_conf['summary_writer']

    manager = Manager()
    episode = manager.dict()
    info = manager.dict()
    managed_memory = dict(
        episode=episode,
        info=info,
    )
    parallel_train_manager = ParallelTrainManager(managed_memory=managed_memory,
                                                  summary_writer=basic_conf['summary_writer'],
                                                  jobdir='/tmp/test')

    print([parallel_train_manager.add_module(task_fn) for task_fn in tasks_fn])
    print(parallel_train_manager.add_module(forward_model_fn))
    print([parallel_train_manager.add_module(policy_fn) for policy_fn in policies_fn])
    print(parallel_train_manager.add_module(task_selector_fn))
    print(parallel_train_manager.add_module(task_planner_fn))
    print('Done')
