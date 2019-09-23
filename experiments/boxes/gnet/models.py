import numpy as np

from gnet.models import GnetModel

class Random(GnetModel):
    def __init__(self, parent, inputs, normalizer, tasks_specs, task_from_id, task_to_id, scope='simple'):
        super().__init__(parent=parent, inputs=inputs, normalizer=normalizer, tasks_specs=tasks_specs,
                         task_from_id=task_from_id, task_to_id=task_to_id, scope=scope)

    def sample_goal(self, o, constant_coords, variable_coords):
        r = np.random.uniform(5, 10)
        phi = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        return np.asarray([x, y])
