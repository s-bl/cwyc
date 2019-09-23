import numpy as np

from parallel_module import ParallelModule
from simple_buffer import SimpleConsecutiveBuffer
from forward_model.utils import weighted_surprise
from utils import normalize

class TaskSelector(ParallelModule):
    def __init__(self, tasks_specs, surprise_weighting=0.1, buffer_size=100, lr=0.1, reg=1e-3, precision=1e-3,
                 eps_greedy_prob=0.05, surprise_hist_weighting=0.99, epsilon=0.1, fixed_Q=None, scope='TaskSelector'):
        super().__init__(scope)
        self._tasks_specs = tasks_specs
        self._num_tasks = len(self._tasks_specs)
        self._surprise_weighting = surprise_weighting
        self._lr = lr
        self._reg = reg
        self._precision = precision
        self._eps_greedy_prob = eps_greedy_prob
        self._surprise_hist_weighting = surprise_hist_weighting
        self._epsilon = epsilon
        self._buffer = SimpleConsecutiveBuffer({
                'abs_success_rates_deriv':(self._num_tasks,),
                'surprises': (self._num_tasks,),
            },
        buffer_size
        )
        self._Q = np.zeros(self._num_tasks)
        self._fixed_Q = fixed_Q

    def store_transitions(self, batch, info):
        if self._fixed_Q is None:
            new_batch = dict(
                abs_success_rates_deriv=np.asarray(info['abs_success_rates_deriv']),
                surprises=np.asarray([np.max(surprises, axis=0) for surprises in batch['surprises']]),
            )

            self._buffer.store_transitions(new_batch)

    def train(self):
        if self._fixed_Q is None:
            batch = self._buffer.sample(self._buffer.get_current_size())
            batch_len = self._buffer.get_current_size()

            # Normalize absolute success rate derivative between tasks
            r_abs_success_rate_deriv = normalize(batch['abs_success_rates_deriv'][-1], zero_out=True)

            # Find last surprise signal and weight according to how long it is in the past
            w_surprise = weighted_surprise(batch['surprises'], self._surprise_hist_weighting, batch_len)
            r_surprise = self._surprise_weighting * w_surprise
            r  =  r_abs_success_rate_deriv + r_surprise
            self._Q += self._lr * (r - self._Q)
            self._Q = self._Q * (self._Q > self._precision).astype(np.int32)

        out = dict()
        if self._fixed_Q is not None:
            Q = self._fixed_Q
        else:
            Q = self._Q + self._epsilon

            out.update({f'task_{i}/abs_success_deriv': ('scalar', r_abs_success_rate_deriv[i]) for i in range(self._num_tasks)})
            out.update({f'task_{i}/surprise': ('scalar', r_surprise[i]) for i in range(self._num_tasks)})

        out.update({f'task_{i}/Q': ('scalar', Q[i]) for i in range(self._num_tasks)})
        out.update({f'task_{i}/prob': ('scalar', p) for i, p in enumerate(normalize(Q))})

        return out

    def sample(self):
        if self._fixed_Q is not None:
            Q = self._fixed_Q
        else:
            Q = self._Q + self._epsilon

        if np.random.rand() < self._eps_greedy_prob and self._fixed_Q is None: # act randomly
            return np.random.randint(0, self._num_tasks)
        else:
            p = normalize(Q)
            return np.random.choice(self._num_tasks, p=p)

    def get_global_params(self, scope=''):
        return (self._scope, [('Q', self._Q)])

    def get_params(self, scope=''):
        return self.get_global_params(scope)

    def set_global_params(self, params, scope=''):
        params = [param[1] for param in params]
        self._Q = params[0]

    def set_params(self, params, scope=''):
        self.set_global_params(params, scope)

if __name__ == '__main__':
    from config import basic_configure

    config = basic_configure()

    task_selector = config['task_selector_fn'][0](**config['task_selector_fn'][1])

    '''
    import dill
    from tensorboardX import SummaryWriter

    summary_writer = SummaryWriter('/tmp/tf_board')

    tasks_specs = [
        dict(
            id=0,
            mg=range(0,2),
            g_dim=2,
            th=0.5,
        ),
        dict(
            id=1,
            mg=range(2,4),
            g_dim=2,
            th=0.5
        )
    ]

    task_selector = TaskSelector(tasks_specs, 0.1)

    with open('/tmp/episode.pkl', 'rb') as f:
        episode, info = dill.load(f)

    info['abs_success_rates_deriv'] = info['success_rates_deriv']

    for _ in range(1000):
        if _ == 100:
            episode['surprises'][1][10,1] = 1
        task_selector.store_transitions(episode, info)
        res = task_selector.train()
        for key, value in res.items():
            summary_writer.add_scalar(key, value, _)
        if _ == 100:
            episode['surprises'][1][10,1] = 0
    '''
