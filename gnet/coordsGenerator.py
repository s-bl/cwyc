import numpy as np

from simple_buffer import SimpleConsecutiveBuffer

class CoordsGenerator:
    def __init__(self, o_dim, mo, excluded_coords=[], buffer_size=int(1e5), th=.5):
        self.varCoords = []
        self._excluded_coords = excluded_coords
        self._buffer = SimpleConsecutiveBuffer(dict(
            o=(o_dim,),
        ), size=buffer_size)
        self._mo = mo
        self._th = th
        self._o_dim = o_dim

        self._cov = np.zeros((o_dim, o_dim))

    def store_transitions(self, batch):
        self._buffer.store_transitions(batch)

    def update(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            obs = self._buffer._buffers['o'][:self._buffer.get_current_size()]
            self._cov = np.nan_to_num(np.corrcoef(obs.T))
        self.varCoords = list(set(np.nonzero(np.abs(self._cov)[self._mo] > self._th)[1]) - set(self._excluded_coords))
