import numpy as np
from parallel_module import ParallelModule
from simple_buffer import SimpleConsecutiveBuffer
from task.utils import mg_fn

class Task(ParallelModule):
    def __init__(self, id, mg, mo, g_dim, th, scope, buffer_size=100, steps_at_target_before_success=5, reward_type='dense',
                 min_num_steps_for_success=0):
        super().__init__(scope)
        self.id = id
        self._mg = mg # positions in goal space arrays
        self._mo = mo # positions on observation vector
        self._g_dim = g_dim
        self._steps_at_target_before_success = steps_at_target_before_success
        self._reward_type = reward_type
        self._min_num_steps_for_success = min_num_steps_for_success
        assert len(self._mg) == self._g_dim
        self._th = th
        self._buffer = SimpleConsecutiveBuffer(
            buffer_shapes=dict(
                success=(1,),
            ),
            size=buffer_size
        )
        self.success_rate = 0
        self.abs_success_rate_deriv = 0

    def store_transitions(self, batch, info):
        new_batch = {'success': np.asarray([info['final_task_success'][i] for i in range(len(info['final_task_success'])) if
                     info['task_seq'][i][-1] == self.id])[:,None]}

        batch_sizes = [value.shape[0] for value in new_batch.values()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        if batch_size == 0: return

        self._buffer.store_transitions(new_batch)

    def train(self):
        if self._buffer.get_current_size() > 0:
            batch = self._buffer.sample(self._buffer._size)
            success_rate = np.mean(batch['success'])
            self.abs_success_rate_deriv = abs(success_rate - self.success_rate)
            self.success_rate = success_rate

        return {'train/success_rate': ('scalar', self.success_rate),
                'train/success_rate_deriv': ('scalar', self.abs_success_rate_deriv)
               }

    def mg_fn(self, x):
        return mg_fn(x, self._mg)

    def assign_goal(self, goal_state, new_task_goal):
        new_goal_state = goal_state.copy()
        new_goal_state[self._mg] = new_task_goal

        return new_goal_state

    def reward_done_success(self, new_ag, new_g, t, T, ag=None, a=None, extract_goal=True):
        if not hasattr(self, '_success_counter'):
            self._success_counter = 0
        if t == 0: self._success_counter = 0
        if extract_goal:
            dist = np.linalg.norm(self.mg_fn(new_ag) - self.mg_fn(new_g))
        else:
            dist = np.linalg.norm(new_ag - new_g)
        # dist = sum((self.mg_fn(new_ag) - self.mg_fn(new_g))**2)**(1/2)
        reward = - 0.3 * dist
        _success = int(dist <= self._th) * (int(t >= self._min_num_steps_for_success) if t is not None else 1)

        if _success:
            self._success_counter += 1
            reward += 10
        else:
            self._success_counter = 0

        success = int(self._success_counter >= self._steps_at_target_before_success)
        # if success: self._success_counter = 0
        done = min(success + int(t==T-1), 1)
        if self._reward_type == 'dense':
            reward += success * 100
        elif self._reward_type == 'sparse':
            reward = success-1
        else:
            raise NotImplementedError

        return np.asarray([reward]), np.asarray([done]), np.asarray([success])

    def get_global_params(self, scope=''):
        return (self._scope, [('success_rate', self.success_rate), ('abs_success_rate_deriv', self.abs_success_rate_deriv)])

    def get_params(self, scope=''):
        return self.get_global_params(scope)

    def set_global_params(self, params, scope=''):
        params = [param[1] for param in params]
        self.success_rate = params[0]
        self.abs_success_rate_deriv = params[1]

    def set_params(self, params, scope=''):
        self.set_global_params(params, scope)

if __name__ == '__main__':
    from config import basic_configure

    basic_config = basic_configure(logdir='/tmp/test', clean=True)

    tasks_fn = basic_config['tasks_fn']

    [task_fn[0](**task_fn[1]) for task_fn in tasks_fn]
