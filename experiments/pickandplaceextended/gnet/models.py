import numpy as np

from gnet.models import GnetModel

class Simple(GnetModel):
    def __init__(self, parent, inputs, normalizer, tasks_specs, task_from_id, task_to_id, scope='simple'):
        super().__init__(parent=parent, inputs=inputs, normalizer=normalizer, tasks_specs=tasks_specs,
                         task_from_id=task_from_id, task_to_id=task_to_id, scope=scope)

    def sample_goal(self, o, constant_coords, variable_coords):
        if self._task_from_id == 0 and self._task_to_id == 1:
            return o[9:12] + np.asarray([0.15,0.0,0.05])
        elif self._task_from_id == 1 and self._task_to_id == 2:
            return o[3:6] - np.asarray([0.52, 0.1, 0.0])
        if self._task_from_id == 1:
            return np.asarray([1,1])
