import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, T, *shape])
                        for key, shape in buffer_shapes.items()}
        self.buffers['ep_T'] = np.empty((self.size, 1), dtype=np.int32)

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

    @property
    def full(self):
        return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        assert self.current_size > 0
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][:self.current_size]

        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_next', 'ag_next'] + list(self.buffers.keys())):
            if key == 'ep_T': continue
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        idxs = self._get_storage_idx(batch_size)

        # load inputs into buffers
        ep_T = np.asarray([ep.shape[0] for ep in list(episode_batch.values())[0]])
        for key in self.buffers.keys():
            if key == 'ep_T':
                self.buffers['ep_T'][idxs] = ep_T[:,None]
            else:
                self.buffers[key][idxs] = 0
                for i, idx in enumerate(idxs):
                    self.buffers[key][idx, :ep_T[i]] = episode_batch[key][i]

    def get_current_episode_size(self):
        return self.current_size

    def get_current_size(self):
        return self.current_size * self.T

    def get_transitions_stored(self):
        return self.n_transitions_stored

    def clear_buffer(self):
        self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        # if inc == 1:
        #     idx = idx[0]
        return idx

    def save(self, path):
        np.save(path, [self.buffers, self.current_size, self.n_transitions_stored])

    def restore(self, path):
        self.buffers, self.current_size, self.n_transitions_stored = np.load(path)
