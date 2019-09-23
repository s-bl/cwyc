import tensorflow as tf
import os
import logging
import click
import numpy as np
import time
import sys
from copy import deepcopy
from collections import deque
import signal

from config import configure_parallel
from interruptHandler import InterruptHandler
from policies.const import TRAIN, EVAL

class Train:
    def __init__(self, scope='train', **kwargs):
        self._scope = scope
        self._ep = 0
        self._env_steps = 0

        self._parallel_rollout_manager, \
        self._parallel_train_manager, \
        self._basic_conf, self._parallel_conf, self._modules_fn, self._managed_memory, \
        self._summary_writer = configure_parallel(**kwargs)
        self._jobdir = self._basic_conf['jobdir']

        self._max_env_steps = int(self._basic_conf['max_env_steps'])
        self._restart_after = self._basic_conf['restart_after']
        self._store_params_every = self._basic_conf['store_params_every']
        self._params_cache_size = self._basic_conf['params_cache_size']
        self._render = self._basic_conf['render']

        self._useSurprise = self._basic_conf['use_surprise']

        self._logger = logging.getLogger(self._scope)

        if self._restart_after is not None and os.path.exists(os.path.join(self._jobdir, 'restart')):
            self._load_train()

        self._cache = []

        self._logger.info(f'Job pid is {os.getpid()}')


    def start_loop(self):
        self._loop()

    def _loop(self):

        if os.path.exists(os.path.join(self._jobdir, 'restart')):
            os.remove(os.path.join(self._jobdir, 'restart'))

        start_time = time.time()

        self._parallel_rollout_manager.set_ep(self._ep)
        self._steps_since_last_write = 0

        # TODO: Remove
        self.debug_buffer = deque(maxlen=1000)

        with InterruptHandler() as h, InterruptHandler(signal.SIGUSR2) as h2:
            while True:
                runtime = time.time() - start_time

                self._logger.info(f'Ep {self._ep}, env steps: {self._env_steps}/{self._max_env_steps}, runtime: {runtime // 60} min'
                                  f'{"" if self._restart_after is None else " (restart in {} min)".format(((int(self._restart_after) * 60) - runtime) // 60)}')
                                  # f' " (restart in "{(runtime - (int(self._restart_after) * 60)) // 60} min)')
                self._parallel_rollout_manager.set_global_params(self._parallel_train_manager.get_global_params())
                self._parallel_rollout_manager.generate_rollouts(render=self._render, mode=TRAIN)

                episode = self._managed_memory['episode']
                self._env_steps += np.sum([ep.shape[0] for ep in episode['o']])
                self._steps_since_last_write += np.sum([ep.shape[0] for ep in episode['o']])

                if not self._useSurprise:
                    self._managed_memory['episode']['surprises'] = [np.zeros_like(surprises) for surprises in self._managed_memory['episode']['surprises']]

                # TODO: Remove
                for i in range(len(self._managed_memory['episode']['surprises'])):
                    if np.any(self._managed_memory['episode']['surprises'][i][:,1:] > 0):
                        self.debug_buffer.append([{key: deepcopy(value[i]) for key, value in self._managed_memory['episode'].items()},
                                                  {key: deepcopy(value[i]) for key, value in self._managed_memory['info'].items()}])
                self._summary_writer.add_scalar('train/debug_buffer_size', len(self.debug_buffer), self._env_steps)

                self._parallel_train_manager.store_transitions()
                self._parallel_train_manager.train()

                if self._steps_since_last_write > 100:
                    self._steps_since_last_write = 0

                    self._parallel_train_manager.write_tflogs(ep=self._ep, env_step=self._env_steps)
                    self._summary_writer.add_scalar('train/ep', self._ep, self._env_steps)

                    if self._basic_conf['eval_runs']:
                        for i in range(len(self._basic_conf['tasks_specs'])):
                            success = 0
                            for _ in range(1):
                                self._parallel_rollout_manager.generate_rollouts(render=self._render, mode=EVAL, final_tasks=i)
                                success += int(self._managed_memory['episode']['success'][0][-1])
                            self._summary_writer.add_scalar(f'Task{i}/eval/success_rate', success/10.0, self._env_steps)

                self._save_params()

                if self._restart_after is not None and runtime >= int(self._restart_after) * 60:
                    self._logger.info(f'Restarting training after {runtime // 60} min after ep {self._ep}')
                    self._save_train()
                    open(os.path.join(self._jobdir, 'restart'), 'w').close()
                    sys.exit(3)

                if h.signal_received:
                    self._save_params(force=True)
                    h.reset()

                if h2.signal_received:
                    np.save(os.path.join(self._jobdir, 'debug_buffer_latest'), self.debug_buffer)
                    h2.reset()

                if self._env_steps >= self._max_env_steps-1:
                    open(os.path.join(self._jobdir, 'done'), 'w').close()
                    break

                self._ep += 1

    def _save_params(self, force=False):
        if self._store_params_every is not None and self._ep % self._store_params_every == 0:
            self._cache.append(self._parallel_train_manager.save_params(f'{self._ep:03}'))
        if force:
            self._cache.append(self._parallel_train_manager.save_global_params('latest'))
            self._parallel_train_manager.save_buffer('latest')
        if len(self._cache) >= self._params_cache_size or force:
            for ep in self._cache:
                for item in ep:
                    np.save(*item)
            self._cache = []

    def _save_train(self):
        if self._ep > 0:
            self._save_params(force=True)
            np.save(os.path.join(self._jobdir, 'train_latest'), [self._ep+1, self._env_steps])

    def _load_train(self):
        if not os.path.exists(os.path.join(self._jobdir, 'restart')): return
        params = np.load(os.path.join(self._jobdir, 'train_latest.npy'))
        self._ep = int(params[0])
        self._env_steps = int(params[1])
        self._logger.info(f'Continue training from ep {self._ep}')

    def __del__(self):
       self._save_train()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


@click.command()
@click.option('--basic_params_path', default=None, help='Path to basic config params')
@click.option('--params-path', default=None, help='Path to stored params')
@click.option('--params-prefix', default=None, help='Params prefix')
@click.option('--clean/--no-clean', default=False, help='Clean job directory')
@click.option('--jobdir', default='/tmp/test', help='Output directory')
@click.option('--max-env-steps', default=None, help='Miximum number of environmental steps')
@click.option('--num-worker', default=None, help='Number of parallel rollout worker')
@click.option('--env', default=None, help='Env specification')
@click.option('--restart-after', default=None, help='Restart training after n minutes')
@click.option('--json-string', default=None, help='Provide json with additional parameter (highest prio)')
@click.option('--experiment', default=None, help='Provide json file with additional params (overrides command line argument)')
@click.option('--seed', default=None, help='Seed for numpy, tensorflow and evs')
@click.option('--store-params-every', default=50, help='Store module params ever n-th episode')
@click.option('--params-cache-size', default=10, help='Number of items stored in cache before writing to disc')
def main(**kwargs):
    train = Train(**kwargs)
    train.start_loop()

if __name__ == '__main__':
    main()
