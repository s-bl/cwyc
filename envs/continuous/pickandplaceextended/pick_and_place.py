import os
import numpy as np
from gym.utils import EzPickle
from gym.envs.robotics import rotations, fetch_env, utils


# Ensure we get the path separator correct on windows
script_path = os.path.dirname(os.path.realpath(__file__))
MODEL_XML_PATH = os.path.join(script_path, 'assets', 'fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, EzPickle):
    def __init__(self, max_episode_steps=50, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'pole0:joint': [1.55, 0.53, 0.4, 1., 0, 0., 0.],
        }

        self.object_x_offset = 0.3
        self.object_y_offset = 0.2

        self.pole_x_offset = 0.15
        self.pole_y_offset = 0.0

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        EzPickle.__init__(self)

        self.max_episode_steps = max_episode_steps

    def set_goal(self, goal, mgs):
        self.goal = goal

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        pole_pos = self.sim.data.get_site_xpos('pole0:site')
        # rotations
        pole_rot = rotations.mat2euler(self.sim.data.get_site_xmat('pole0:site'))
        # velocities
        pole_velp = self.sim.data.get_site_xvelp('pole0:site') * dt
        pole_velr = self.sim.data.get_site_xvelr('pole0:site') * dt
        # gripper state
        pole_rel_pos = pole_pos - grip_pos
        pole_velp -= grip_velp

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.hstack([grip_pos.ravel(), object_pos.ravel(), pole_pos.ravel()])
            # achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), pole_pos.ravel(), pole_rel_pos.ravel(),
            gripper_state, object_rot.ravel(), pole_rot.ravel(),
            object_velp.ravel(), pole_velp.ravel(), object_velr.ravel(), pole_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        while True:
            pole_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                               size=2)
            pole_qpos = self.sim.data.get_joint_qpos('pole0:joint').copy()
            pole_qpos[:2] = pole_xpos
            pole_qpos[0] += self.pole_x_offset
            pole_qpos[1] += self.pole_y_offset

            # Randomize start position of object.
            object_xpos = pole_qpos[:2].copy() + np.asarray([0.1, 0.15]) + self.np_random.uniform(-0.1, 0.1, 2)
            # object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
            #                                                                          size=2)
            # object_xpos[0] += self.object_x_offset
            # object_xpos[1] += self.object_y_offset
            # while np.linalg.norm(object_xpos[0] - pole_qpos[0]) > 0.24 - 0.1 and np.linalg.norm(object_xpos[0] - pole_qpos[0]) < 0.24 + 0.1:
            #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
            #                                                                          size=2)
            #     object_xpos[0] += self.object_x_offset
            #     object_xpos[1] += self.object_y_offset
            if np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) > 0.3: break
        object_qpos = self.sim.data.get_joint_qpos('object0:joint').copy()
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.data.set_joint_qpos('pole0:joint', pole_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            initial_object_position = self.sim.data.get_site_xpos('object0').copy()
            object_goal = initial_object_position[:3] - self.np_random.uniform(0.06, self.target_range, size=3)
            object_goal[2] = initial_object_position[2]
            object_goal[1] = initial_object_position[1]
        gripper_goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        initial_pole_xpos = self.sim.data.get_site_xpos('pole0:site').copy()
        pole_goal = initial_pole_xpos[:3] + self.np_random.uniform(-0.15,0.15,size=3)
        pole_goal[2] = initial_pole_xpos[2]

        if self.has_object:
            goal = np.hstack([gripper_goal.ravel(), object_goal.ravel(), pole_goal.ravel()])
        else:
            goal = gripper_goal.ravel()

        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[3:6] - sites_offset[0]
        site_id = self.sim.model.site_name2id('target:gripper')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[1]
        site_id = self.sim.model.site_name2id('target:pole')
        self.sim.model.site_pos[site_id] = self.goal[6:] - sites_offset[2]
        self.sim.forward()
