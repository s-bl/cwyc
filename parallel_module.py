import os
import logging
import numpy as np
import dill
import tensorflow as tf
import gzip

from utils import get_jobdir

class ParallelModule:
    def __init__(self, scope):
        self._scope = scope
        self._logger = logging.getLogger(self._scope)
        self._jobdir = get_jobdir(self._logger)
        self._seed = None

    def get_params(self, scope=''):
        raise NotImplementedError

    def get_global_params(self, scope=''):
        raise NotImplementedError

    def set_params(self, params, scope=''):
        raise NotImplementedError

    def set_global_params(self, params, scope=''):
        raise NotImplementedError

    def store_transitions(self, batch, info):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_params(self, path, prefix=''):
        prefix = '' if not prefix else f'_{prefix}'
        # np.save(os.path.join(path, f'{self._scope}{prefix}'), self.get_params())
        return (os.path.join(path, f'{self._scope}{prefix}'), self.get_params())

    def save_global_params(self, path, prefix=''):
        prefix = '' if not prefix else f'_{prefix}'
        # np.save(os.path.join(path, f'{self._scope}{prefix}'), self.get_global_params())
        return (os.path.join(path, f'{self._scope}{prefix}'), self.get_global_params())

    def load_params(self, path, prefix=''):
        prefix = '' if not prefix else f'_{prefix}'
        params = np.load(os.path.join(path, f'{self._scope}{prefix}.npy'))[1]
        self.set_params(params)

    def load_global_params(self, path, prefix=''):
        prefix = '' if not prefix else f'_{prefix}'
        params = np.load(os.path.join(path, f'{self._scope}{prefix}.npy'))[1]
        self.set_global_params(params)

    def save_buffer(self, path, prefix=''):
        if hasattr(self, '_buffer'):
            prefix = '' if not prefix else f'_{prefix}'
            if isinstance(self._buffer, dict):
                for key, buffer in self._buffer.items():
                    buffer.save(os.path.join(path, f'{self._scope}_buffer_{key}{prefix}'))
            else:
                self._buffer.save(os.path.join(path, f'{self._scope}_buffer{prefix}'))

    def restore_buffer(self, path, prefix=''):
        if hasattr(self, '_buffer'):
            prefix = '' if not prefix else f'_{prefix}'
            if isinstance(self._buffer, dict):
                for key, buffer in self._buffer.items():
                    buffer.restore(os.path.join(path, f'{self._scope}_buffer_{key}{prefix}'))
            else:
                self._buffer.restore(os.path.join(path, f'{self._scope}_buffer{prefix}'))

    def set_seed(self, seed):
        self._seed = seed
        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)


