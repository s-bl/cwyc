import numpy as np

class EpisodicBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T):

        self._buffer_shapes = buffer_shapes
        self._size = size_in_transitions // T
        self._T = T

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self._buffers = {key: np.empty([self._size, T, *shape])
                         for key, shape in buffer_shapes.items()}
        self._buffers['ep_T'] = np.empty((self._size, 1), dtype=np.int32)

        # memory management
        self._current_size = 0

    @property
    def full(self):
        return self._current_size == self._size

    def sample_episodes(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        idxs = np.random.randint(0, self._current_size, batch_size)

        assert self._current_size > 0
        for key in self._buffers.keys():
            buffers[key] = self._buffers[key][idxs]

        return buffers

    def sample_transitions(self, batch_size, hist_length=1):
        episodes = self.sample_episodes(batch_size)
        buffers = {}

        idxs = np.asarray([np.random.randint(hist_length, ep_T) for ep_T in episodes['ep_T']])

        for key in episodes.keys():
            if key == 'ep_T':
                buffers[key] = episodes[key]
            else:
                buffers[key] = np.empty((batch_size, hist_length, *self._buffer_shapes[key]))
                for i, idx in enumerate(idxs):
                    buffers[key][i] = episodes[key][i, idx-hist_length:idx]

        return buffers

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        idxs = self._get_storage_idx(batch_size)

        # load inputs into buffers
        ep_T = np.asarray([ep.shape[0] for ep in list(episode_batch.values())[0]])
        for key in self._buffers.keys():
            if key == 'ep_T':
                self._buffers['ep_T'][idxs] = ep_T[:,None]
            else:
                self._buffers[key][idxs] = 0
                for i, idx in enumerate(idxs):
                    self._buffers[key][idx, :ep_T[i]] = episode_batch[key][i]

        return idxs

    def get_current_episode_size(self):
        return self._current_size

    def get_current_size(self):
        return self._current_size * self._T

    def clear_buffer(self):
        self._current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self._size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self._current_size+inc <= self._size:
            idx = np.arange(self._current_size, self._current_size + inc)
        elif self._current_size < self._size:
            overflow = inc - (self._size - self._current_size)
            idx_a = np.arange(self._current_size, self._size)
            idx_b = np.random.randint(0, self._current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self._size, inc)

        # update replay size
        self._current_size = min(self._size, self._current_size + inc)

        return idx

    def save(self, path):
        np.save(path, [self._buffers, self._current_size])

    def restore(self, path):
        self._buffers, self._current_size = np.load(path)

class SimpleBuffer:
    def __init__(self, buffer_shapes, size):
        self._buffer_shapes = buffer_shapes
        self._size = size

        self._buffers = {key: np.empty([self._size, *shape])
                        for key, shape in buffer_shapes.items()}

        self._current_size = 0

    @property
    def full(self):
        return self._current_size == self.size

    def sample(self, batch_size):
        buffers = {}

        idxs = np.random.randint(0, self._current_size, batch_size)

        assert self._current_size > 0
        for key in self._buffers.keys():
            buffers[key] = self._buffers[key][idxs]

        return buffers

    def store_transitions(self, transitions_batch):

        batch_sizes = [value.shape[0] for value in transitions_batch.values()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        idxs = self._get_storage_idx(batch_size)

        for key in self._buffers.keys():
            self._buffers[key][idxs] = transitions_batch[key]

    def get_current_size(self):
        return self._current_size

    def clear_buffer(self):
        self._current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self._size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self._current_size+inc <= self._size:
            idx = np.arange(self._current_size, self._current_size+inc)
        elif self._current_size < self._size:
            overflow = inc - (self._size - self._current_size)
            idx_a = np.arange(self._current_size, self._size)
            idx_b = np.random.randint(0, self._current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self._size, inc)

        # update replay size
        self._current_size = min(self._size, self._current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx

    def save(self, path):
        np.save(path, [self._buffers, self._current_size])

    def restore(self, path):
        self._buffers, self._current_size = np.load(path)

class SimpleConsecutiveBuffer(SimpleBuffer):
    def __init__(self, buffer_shapes, size):
        super().__init__(buffer_shapes=buffer_shapes, size=size)

        self._last_idx = 0

    def sample(self, batch_size):
        buffers = {}

        if self._current_size < self._size or self._last_idx > batch_size:
            idxs = np.arange(max(0, self._last_idx-batch_size), self._last_idx)
        else:
            underflow =  batch_size - self._last_idx
            idx_a = np.arange(self._size-underflow, self._size)
            idx_b = np.arange(0, self._last_idx)
            idxs = np.concatenate([idx_a, idx_b])

        assert self._current_size > 0
        for key in self._buffers.keys():
            buffers[key] = self._buffers[key][idxs]

        return buffers

    def store_transitions(self, transitions_batch):

        batch_sizes = [value.shape[0] for value in transitions_batch.values()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        idxs = self._get_storage_idx(batch_size)

        for key in self._buffers.keys():
            self._buffers[key][idxs] = transitions_batch[key]

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self._size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self._last_idx+inc <= self._size:
            idx = np.arange(self._last_idx, self._last_idx+inc)
        else:
            overflow = inc - (self._size - self._last_idx)
            idx_a = np.arange(self._last_idx, self._size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])

        # update replay size
        self._current_size = min(self._size, self._current_size+inc)

        self._last_idx = idx[-1] + 1

        if inc == 1:
            idx = idx[0]
        return idx

    def save(self, path):
        np.save(path, [self._buffers, self._current_size, self._last_idx])

    def restore(self, path):
        self._buffers, self._current_size, self._last_idx = np.load(path)

if __name__ == '__main__':
    episodicBuffer = EpisodicBuffer(dict(o=(5,)), 10**6, T=500)
    episodicBuffer.store_episode(dict(o=[np.random.rand(10,5), np.random.rand(15,5)]))
    print(episodicBuffer.sample_transitions(2, hist_length=2)['o'])
    print('done')
    # consecutiveBuffer = SimpleConsecutiveBuffer(dict(success=(1,)), size=100)
    # consecutiveBuffer.store_transitions(dict(success=np.zeros((95,1))))
    # consecutiveBuffer.store_transitions(dict(success=np.ones((5,1))))
    # print(consecutiveBuffer._last_idx)
    # print(consecutiveBuffer.sample(10))
    # consecutiveBuffer.store_transitions(dict(success=np.ones((1,1))*5))
    # print(consecutiveBuffer._last_idx)
    # print(consecutiveBuffer.sample(5))
    # print(consecutiveBuffer.sample(30))


    # consecutiveBuffer.store_transitions(dict(success=np.random.randint(0,1,(60,1))))
    # consecutiveBuffer.store_transitions(dict(success=np.ones((5,1))))
    # print(consecutiveBuffer.get_current_size())
    # print(consecutiveBuffer._last_idx)
    # print(consecutiveBuffer.sample(batch_size=10))

