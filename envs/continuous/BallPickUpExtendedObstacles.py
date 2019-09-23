import numpy as np
import os
from gym import utils as utils
from gym import spaces
import gym
from gym import error
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: "
        "https://github.com/openai/mujoco-py/.)".format(
            e))


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape

    return np.linalg.norm(goal_a - goal_b)


def to_rgb(h):
    return tuple('{:.2}'.format((int(h[j:j + 2], 16) / 255.)) for j in (0, 2, 4))


colors_hex = [
    "5e81b5",
    "e19c24",
    "8fb032",
    "eb6235",
    "8778b3",
    "c56e1a",
]

colors_rgb = [' '.join(to_rgb(h)) for h in colors_hex]

dynamic_color = "0.5 0.5 0.5"
prob_color = colors_rgb[1]
magic_color = colors_rgb[3]

masses = [0.01, 0.01, 0.01, 0.01, 0.01]
dampings = [0.5, 0.5, 0.5, 0.5, 0.5]

start_pos = [

    [0, 0, 2.0]

]

PROB_NAME = 'probBox'
DYNAMIC_NAME = 'dynamicBox'
MAGIC_NAME = 'magicBox'
RANDOM_WALK = 'randomWalkBox'


def prob_body_xml(name, pos, size='0.6 0.6 0.6'):
    color = prob_color

    return f""" <body name="{name}" pos="{pos}" >
                <geom name="{name}:geom" type="box" size="{size}" rgba=" """ + color + f""" 1" contype="1" 
                conaffinity="1"  mass="10000" density="0"></geom>
                </body>
                <body name="{name}Goal" pos="0 0 0.1">
                <geom name="{name}Goal:geom" type="cylinder" size="1 0.01" rgba=" """ + color + """ 0.0" density="0" 
                contype="0" conaffinity="0"></geom>
                </body>
            """


def dynamic_body_xml(name, pos, size='0.6 0.6 0.6'):
    return f""" <body name="{name}" pos="{pos}" >
            <geom name="{name}:geom" type="box" size="{size}" rgba=" """ + colors_rgb[
        1] + f""" 1" contype="1" conaffinity="1"  mass="100" density="0"></geom>
            <joint armature="0" name="physicalBox:slidex" type="slide" pos="0 0 0" axis="1 0 0" limited="false" 
            damping="{
    dampings[0]}"stiffness="0.1"/>
            <joint armature="0" name="physicalBox:slidey" type="slide" pos="0 0 0" axis="0 1 0" limited="false" 
            damping="{
    dampings[0]}" stiffness="0.1"/>
            </body>
        """


def random_walk_body_xml(name, pos, size='0.6 0.6 0.6'):
    color = dynamic_color
    return f""" <body name="{name}" pos="{pos}" >
            <geom name="{name}:geom" type="box" size="{size}" rgba=" """ + color + f""" 1" contype="1" conaffinity="1"  
            mass="10" density="0"></geom>
            <joint armature="0" name="{name}:slidex" type="slide" pos="0 0 0" axis="1 0 0" limited="false" damping="{
    dampings[0]}"stiffness="0.1"/>
            <joint armature="0" name="{name}:slidey" type="slide" pos="0 0 0" axis="0 1 0" limited="false" damping="{
    dampings[0]}" stiffness="0.1"/>
            </body>
            <body name="{name}Goal" pos="0 0 0.1">
            <geom name="{name}Goal:geom" type="cylinder" size="1 0.01" rgba=" """ + color + """ 0.0" density="0" 
            contype="0" conaffinity="0"></geom>
            </body>
        """


def magic_body_xml(name, pos, size='0.6 0.6 0.6'):
    color = magic_color
    return f"""   <body name="{name}" pos="{pos}" >
                <geom name="{name}:geom" type="box" size="{size}" rgba=" """ + color + f""" 1" contype="0" 
                conaffinity="0"  mass="100" density="0"></geom>
                </body>
                <body name="{name}Goal" pos="0 0 0.1">
                <geom name="{name}Goal:geom" type="cylinder" size="1 0.01" rgba=" """ + color + """ 0.0" density="0" 
                contype="0" conaffinity="0"></geom>
                </body>
            """


def gen_xml(script_path, playground_dim, num_prob_objects=0, num_random_walk_objects=0, num_dynamic_objects=0,
            num_magic_objects=0, physical_box=True):
    world_body_xml = f"""
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" 
    specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba=".85 .9 .9 1" size="40 40 40" 
    type="plane"/>
    <body name="playground" pos="0 0 1">
      <geom name="playground:north" pos="0 {playground_dim} 0" size="{playground_dim} 0.2 2" type="box" 
      rgba=".86 .81 .7 1" conaffinity="1" contype="1"></geom>
      <geom name="playground:west" pos="-{playground_dim} 0 0" size="0.2 {playground_dim} 2" type="box" 
      rgba=".86 .81 .7 1" conaffinity="1" contype="1"></geom>
      <geom name="playground:east" pos="{playground_dim} 0 0" size="0.2 {playground_dim} 2" type="box" 
      rgba=".86 .81 .7 1" conaffinity="1" contype="1"></geom>
      <geom name="playground:south" pos="0 -{playground_dim} 0" size="{playground_dim} 0.2 2" type="box" 
      rgba=".86 .81 .7 1" conaffinity="1" contype="1"></geom>
    </body>
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso:geom" pos="0 0 0" size="0.5" type="sphere" rgba=" """ + colors_rgb[3] + """  1" conaffinity="1" 
      condim="3" />
      <joint armature="0" name="torso:slidex" type="slide" pos="0 0 0" axis="1 0 0" limited="false" damping="0" 
      stiffness="0"/>
      <joint armature="0" name="torso:slidey" type="slide" pos="0 0 0" axis="0 1 0" limited="false" damping="0" 
      stiffness="0"/>
      <!--
      <site name="torso:site" type="ellipsoid" size="0.8 0.8 0.8" rgba="0.2 0.9 0.2 1" />
      -->
    </body>
    
    <body name="pickupBox" pos="0 0 .7">
        <geom name="pickupBox:geom" type="box" size="0.6 0.6 0.6" rgba=" """ + colors_rgb[4] + """ 1" contype="0" 
        conaffinity="0" density="0"></geom>
    </body>
    <body name="pickupBoxGoal" pos="0 0 0.1">
        <geom name="pickupBoxGoal:geom" type="cylinder" size="1 0.01" rgba=" """ + colors_rgb[4] + """ 0.5" density="0" 
        contype="0" conaffinity="0"></geom>
    </body>
    <body name="locomotionGoal" pos="0 0 .01">
        <geom name="pos_goal:geom" type="cylinder" size="1 0.01" rgba=" """ + colors_rgb[3] + """ 0.5" density="0" 
        contype="0" conaffinity="0"></geom>
    </body>

    """
    if physical_box:
        world_body_xml += """"
        <body name="physicalBox" pos="0 0 .7">
        <geom name="physicalBox:geom" type="box" size="0.6 0.6 0.6" rgba=" """ + colors_rgb[2] + """ 1" contype="1" 
        conaffinity="1" mass="10000"></geom>
        <!-- <geom name="physicalBox:geom" type="box" size="0.6 0.6 0.6" rgba=" """ + colors_rgb[
            2] + """ 1" contype="1" conaffinity="1" mass=" """ + str(masses[0]) + """ "></geom>
        <joint armature="0" name="physicalBox:slidex" type="slide" pos="0 0 0" axis="1 0 0" limited="false" 
        damping=" """ + str(
            dampings[0]) + """ " stiffness="0"/>
        <joint armature="0" name="physicalBox:slidey" type="slide" pos="0 0 0" axis="0 1 0" limited="false" 
        damping=" """ + str(
            dampings[0]) + """ " stiffness="0"/> -->
    </body>
    <body name="physicalBoxGoal" pos="0 0 0.1">
        <geom name="physicalBoxGoal:geom" type="cylinder" size="1 0.01" rgba=" """ + colors_rgb[2] + """ 0.5" 
        density="0" contype="0" conaffinity="0"></geom>
    </body>"
    """

    for j in range(num_prob_objects):
        world_body_xml += prob_body_xml(f"{PROB_NAME}{j}", "1 1 0.6")

    for j in range(num_random_walk_objects):
        world_body_xml += random_walk_body_xml(f"{RANDOM_WALK}{j}", "1 1 0.6")

    for j in range(num_dynamic_objects):
        world_body_xml += dynamic_body_xml(f"{DYNAMIC_NAME}{j}", "1 1 0.6")

    for j in range(num_magic_objects):
        world_body_xml += magic_body_xml(f"{MAGIC_NAME}{j}", "1 1 0.6")

    xml = f"""<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true" texturedir="{script_path}/assets/textures/" />
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" 
    rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <!-- <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/> -->
    <texture type="2d" file="woodenfloor_light.png" name="texplane"/>
    <material name="MatPlane" reflectance="0" shininess="0" specular="0" texrepeat="10 10" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    {world_body_xml}
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="torso:slidex" gear="1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="torso:slidey" gear="1"/>
  </actuator>
  <!--
  <sensor>
    <touch site="torso:site" />
  </sensor>
  -->
</mujoco>"""

    return xml


def compute_distance(a, b):
    assert a.shape == b.shape

    return np.linalg.norm(a - b)


class Ball(gym.GoalEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', max_episode_steps=400, goal_max_dist=10.0, goal_min_dist=2.0,
                 max_dist_boxes=5., min_dist_boxes=1.,
                 distance_threshold=1.2, relative_goals=False,
                 num_prob_objects=0,  # Can be picked up only with prop 'see next parameter
                 prob_objects_prob=(),  # length must match value of num_prob_objects
                 num_random_walk_objects=0,
                 num_magic_objects=0,
                 num_dynamic_objects=0,
                 physical_box=True,
                 init_pos_noise=.1,
                 ):

        self.physical_box = physical_box

        self.relative_goals = relative_goals

        # Distance in which goals are spawned
        self.goal_max_dist = goal_max_dist
        self.goal_min_dist = goal_min_dist

        # Threshold for reaching goals and picking up objects
        self.distance_threshold = distance_threshold

        self.dofs = 2  # x,y-plane

        # Initial noise in position of agent
        self.init_pos_noise = init_pos_noise

        # Distance in which boxes are spawned
        self.max_dist_boxes = max_dist_boxes
        self.min_dist_boxes = min_dist_boxes

        # Number of episode steps
        self._max_episode_steps = max_episode_steps

        self.min_dist_between_boxes = 0

        self.loco_goal = 0

        self.has_pickupbox = 0
        self.has_physical_box = 0

        self.num_prob_objects = num_prob_objects
        # by default prob objects can't be picked up
        prob_objects_prob = np.ones(num_prob_objects) if len(prob_objects_prob) == 0 else prob_objects_prob

        self.num_dynamic_objects = num_dynamic_objects
        self.num_random_walk_objects = num_random_walk_objects
        self.num_magic_objects = num_magic_objects

        assert self.num_prob_objects == len(prob_objects_prob)

        self.prob_objects_prob = {PROB_NAME + str(j): prob_objects_prob[j] for j in range(self.num_prob_objects)}

        self.random_walk_boxes = [RANDOM_WALK + str(j) for j in range(self.num_random_walk_objects)]
        self.pickupable_boxes = ['pickupBox']
        if physical_box:
            self.pickupable_boxes += ['physicalBox']
        self.box_names = self.pickupable_boxes + [PROB_NAME + str(i) for i in range(self.num_prob_objects)] + \
                         [DYNAMIC_NAME + str(i) for i in range(self.num_dynamic_objects)] + [MAGIC_NAME + str(i) for i
                                                                                             in range(
                self.num_magic_objects)] + \
                         self.random_walk_boxes

        self.num_boxes = len(self.box_names)

        self.goal_names = ['locomotionGoal'] + [box + 'Goal' for box in self.box_names]

        self.can_pickup = np.concatenate(
            [np.ones(2), np.zeros(self.num_prob_objects), np.zeros(self.num_dynamic_objects),
             np.zeros(self.num_magic_objects), np.zeros(self.num_random_walk_objects)])
        self.has_box = np.zeros(len(self.box_names))

        # +1 because of locomotion goal
        self.subgoal_indices = np.arange((self.num_boxes + 1) * self.dofs).reshape(-1, self.dofs)

        # contains a list of dependencies for each goal
        # it is a list because possibly 1 goal can have multiple mutually exclusive dependencies??
        self.goal_dependency = np.arange(len(self.goal_names)).tolist()
        self.goal_dependency[1] = 0
        self.goal_dependency[2] = 1
        for j, name in enumerate(self.goal_names):
            if 'magic' in name:
                self.goal_dependency[j] = 0

        # define subgoal bitmasks
        self.subgoal_masks = np.zeros((self.num_boxes + 1, (self.num_boxes + 1) * self.dofs))
        for j, idxs in enumerate(self.subgoal_indices):
            self.subgoal_masks[j, idxs] = 1

        script_path = os.path.dirname(os.path.realpath(__file__))
        self._init(reward_type=reward_type, frame_skip=5,
                   xml=gen_xml(num_prob_objects=self.num_prob_objects,
                               num_random_walk_objects=self.num_random_walk_objects,
                               script_path=script_path, physical_box=physical_box,
                               num_magic_objects=self.num_magic_objects,
                               playground_dim=max(self.max_dist_boxes, self.goal_max_dist) + 5))
        utils.EzPickle.__init__(self)

        # override action spaces, because we only want to control the ball
        self.action_space = spaces.Box(low=self.action_space.low[:self.dofs], high=self.action_space.high[:self.dofs],
                                       dtype='float32')

    def _init(self, reward_type, frame_skip, xml):
        self.seed()

        self.reward_type = reward_type
        self.frame_skip = frame_skip

        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.goal = self._sample_goal()
        self.init_goal = self.goal.copy()

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation['observation'].size
        self.g_dim = observation['desired_goal'].size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype='float32')

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Dict({
            'desired_goal': spaces.Box(low[:self.g_dim], high[:self.g_dim], dtype='float32'),
            'achieved_goal': spaces.Box(low[:self.g_dim], high[:self.g_dim], dtype='float32'),
            'observation': spaces.Box(low, high, dtype='float32'),
        })

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)

        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.

        d = goal_distance(achieved_goal, desired_goal)

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _sample_goal(self):
        return self._sample_goal_positions().flatten()

    def set_goal(self, goal, mgs):
        self.goal = goal
        for j, goal_name in enumerate(self.goal_names):
            pg_idx = self.model.body_name2id(goal_name)
            self.model.body_pos[pg_idx, 0:self.dofs] = goal[j*self.dofs:(j+1)*self.dofs]

    def _random_walk(self, qvel):

        indices = self.random_walk_indices
        num_random_walk = len(self.random_walk_boxes)
        # repeat because of each degree of freedom
        has_box = np.repeat(self.has_box[-num_random_walk:], self.dofs)
        delta = np.ones_like(indices) * .01 * np.random.choice([-1, 1], len(indices))
        qvel[indices] += delta
        qvel[indices] *= (1 - has_box)
        return qvel

    def _sample_object_positions(self):
        pos = [[0, 0]]
        for _ in self.box_names:
            pos_arr = np.asarray(pos)
            while True:
                r = self.np_random.uniform(self.min_dist_boxes, self.max_dist_boxes)
                phi = self.np_random.uniform(0, 2 * np.pi)
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                if np.all(np.linalg.norm(pos_arr - np.asarray([[x, y]]), axis=1) > 1.2):
                    pos.append([x, y])
                    break
        return pos[1:]

    @property
    def _box_positions(self):
        positions = []
        for box in self.box_names:
            if 'randomWalkBox' in box:
                jnt_idx = self.sim.model.joint_name2id(box + ':slidex')
                jnt_pos = self.sim.data.qpos[jnt_idx:jnt_idx+self.dofs]
                mo_idx = self.model.body_name2id(box)
                mo_pos = self.model.body_pos[mo_idx][:2]
                positions.append(jnt_pos + mo_pos)
            else:
                mo_idx = self.model.body_name2id(box)
                # get x,y positions
                positions.append(self.model.body_pos[mo_idx][:2])

        # return 2D array
        return np.asarray(positions)

    @property
    def _goal_positions(self):
        # returns it in the same order as box positions
        positions = []
        for goal in self.goal_names:
            mo_idx = self.model.body_name2id(goal)
            # get x,y positions
            positions.append(self.model.body_pos[mo_idx][:2])

        # return 2D array
        return np.asarray(positions)

    def _sample_goal_positions(self, min_dist=1.5):
        # generate goal for locomotion + each box
        pos = self._box_positions.tolist()
        goals = []
        for _ in ['Locomotion'] + self.box_names:
            pos_arr = np.asarray(pos)
            while True:
                r = self.np_random.uniform(self.min_dist_boxes, self.max_dist_boxes)
                phi = self.np_random.uniform(0, 2 * np.pi)
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                if np.all(np.linalg.norm(pos_arr - np.asarray([[x, y]]), axis=1) > min_dist):
                    pos.append([x, y])
                    goals.append([x, y])
                    break

        return np.asarray(goals)

    def reset_model(self):

        qpos = self.init_qpos.copy() + self.np_random.uniform(size=self.model.nq, low=-self.init_pos_noise,
                                                              high=self.init_pos_noise)
        qvel = self.init_qvel.copy() + self.np_random.randn(self.model.nv) * .1

        self.has_box = np.zeros_like(self.box_names)

        random_box_positions = self._sample_object_positions()

        # init box positions
        for j, ((x, y), box) in enumerate(zip(random_box_positions, self.box_names)):
            mo_idx = self.model.body_name2id(box)
            self.model.body_pos[mo_idx][:2] = [x, y]
            self.model.body_pos[mo_idx][2] = 0.75
            ge_idx = self.model.geom_name2id(f'{box}:geom')
            if 'physical' in box or 'prob' in box:
                if 'prob' in box:
                    self.can_pickup[j] = self.np_random.rand() > self.prob_objects_prob[box]
                if not self.can_pickup[j] or 'physical' in box:
                    self.model.geom_contype[ge_idx] = self.model.geom_conaffinity[ge_idx] = 1
                    self.model.body_mass[mo_idx] = 100000
                else:
                    self.model.geom_contype[ge_idx] = self.model.geom_conaffinity[ge_idx] = 0
                    self.model.body_mass[mo_idx] = 0

        # init goal positions
        self.goals = self._sample_goal_positions()
        self.goal = self.goals.flatten()
        self.init_goal = self.goal.copy()

        for (xg, yg), box in zip(self.goals[1:], self.box_names):
            mo_idx = self.model.body_name2id(box + 'Goal')
            self.model.body_pos[mo_idx][:2] = [xg, yg]
            self.model.body_pos[mo_idx][2] = 0.02

        # init locomotion goal position
        xg, yg = self.goals[0]
        mo_idx = self.model.body_name2id('locomotionGoal')
        self.model.body_pos[mo_idx][:2] = [xg, yg]
        self.model.body_pos[mo_idx][2] = 0.02

        self.has_box = np.zeros(len(self.box_names))

        # get random walk indices
        self.random_walk_indices = []
        for box in self.random_walk_boxes:
            joint_id = self.model.joint_name2id(box + ':slidex')
            self.random_walk_indices.extend(np.arange(joint_id, joint_id + self.dofs))

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):

        box_positions = self._box_positions.flatten()
        # print(self.sim.data.qpos.flat.shape, self.sim.data.qpos.flat.shape, box_positions.shape)
        obs = np.concatenate([
            self.sim.data.qpos[:2].flat,
            box_positions,
            self.sim.data.qvel.flat,
            self.has_box
        ])

        achieved_goal = np.zeros((1 + len(self.box_names), self.dofs), dtype=np.float32)

        # achieved goal for locomotion task
        achieved_goal[0, :] = self.sim.data.qpos.flatten()[0:self.dofs].copy()
        if self.relative_goals:
            achieved_goal[0, :] -= self.sim.data.qpos.flat[0:self.dofs]

        # achieved goal for different boxes
        achieved_goal[1:] = self._box_positions
        desired_goal = np.asarray(self.goal)
        if self.relative_goals:
            desired_goal -= np.tile(self.sim.data.qpos.flat[0:self.dofs], 3)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.flatten().copy(),
            'desired_goal': desired_goal.copy(),
        }

    def get_positions(self):
        box_ids = [self.model.body_name2id(name) for name in self.box_names]
        box_positions = np.vstack([self.model.body_pos[idx][:self.dofs].flat for idx in box_ids])
        return box_positions

    def _random_pos(self):
        # get random position at a minimal distance from other objects
        box_positions = self._box_positions
        goal_position = self._goal_positions

        # the sampled position mus not be close to any of these positions
        positions = np.vstack([box_positions, goal_position])

        while True:
            r = self.np_random.uniform(self.min_dist_boxes, self.max_dist_boxes)
            phi = self.np_random.uniform(0, 2 * np.pi)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            if np.all(np.linalg.norm(positions - np.asarray([[x, y]]), axis=1) > 1.2):
                return x, y

    def _pickup(self, pos_self_before, pos_self_after):
        """
            This part determines what happens when the ball
            comes near a box
        """
        pos = []
        for i, name in enumerate(self.box_names):
            mo_idx = self.model.body_name2id(name)
            ge_idx = self.model.geom_name2id(name + ':geom')
            xy = self.model.body_pos[mo_idx][:2]
            pos.append(xy)
            if compute_distance(pos_self_before, xy) <= self.distance_threshold:
                if name == 'physicalBox':
                    self.has_box[i] = 1 if self.has_box[i - 1] == 1 else 0
                elif 'magic' in name:
                    # spawn the box somewhere else, impossible box
                    xn, yn = self._random_pos()
                    self.model.body_pos[mo_idx][:2] = xn, yn
                    self.has_box[i] = 0
                elif self.has_box[i] == 0:
                    self.has_box[i] = 1

            if self.can_pickup[i] == 1 and self.has_box[i] == 1:
                assert 'magic' not in name
                self.model.geom_contype[ge_idx] = self.model.geom_conaffinity[ge_idx] = 0
                # set z axis
                self.model.body_pos[mo_idx][2] = np.sum(self.can_pickup[0:i]) * 2 + 2.0
                self.model.body_pos[mo_idx][:2] = pos_self_after.copy()
                if name == 'pickupBox' and self.physical_box:
                    # make phyiscal box enetrable
                    physical_ge_idx = self.model.geom_name2id('physicalBox:geom')
                    self.model.geom_contype[physical_ge_idx] = self.model.geom_conaffinity[physical_ge_idx] = 0

    def do_simulation(self, ctrl, n_frames):
        """
            Need to override this method because there are
            other joints to control in the environment (moving boxes).
        """
        self.sim.data.ctrl[:2] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def step(self, a):
        pos_self_before = self.sim.data.qpos.flatten()[0:self.dofs].copy()

        self.do_simulation(a, self.frame_skip)

        # Implement pickup box mechanism
        pos_self_after = self.sim.data.qpos.flatten()[0:self.dofs].copy()

        qvel = self.sim.data.qvel.flatten()

        # apply random walk to random walk indices
        if hasattr(self, 'random_walk_indices') and len(self.random_walk_boxes) > 0:
            qvel = self._random_walk(qvel)

        self._pickup(pos_self_before, pos_self_after)

        # check locomotion goal
        mo_idx = self.model.body_name2id('locomotionGoal')
        if compute_distance(pos_self_before, self.model.body_pos[mo_idx][:2]) <= 1:
            self.loco_goal = 1

        self.set_state(self.sim.data.qpos, qvel)

        done = False
        ob = self._get_obs()

        info = {
            'success': {
                'locomotion': self._is_success(ob['achieved_goal'][:2], ob['desired_goal'][:2])
            }
        }
        for j, box in enumerate(self.box_names):
            info['success'][box] = self._is_success(ob['achieved_goal'][(j+1)*self.dofs:(j+2)*self.dofs],
                                                    ob['desired_goal'][(j+1)*self.dofs:(j+2)*self.dofs])

        reward = 0
        return ob, reward, done, info

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def viewer_setup(self):
        self.viewer.cam.distance = 45
        self.viewer._hide_overlay = True
        # self.viewer.cam.type = const.CAMERA_FIXED
        # self.viewer.cam.fixedcamid = 0
        # self.model.cam_quat[0, 0] = 1.5
        # self.viewer.cam.type = 0
        # self.viewer.cam.trackbodyid = 1

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 800, 600
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

if __name__ == '__main__':
    env = Ball(reward_type="dense",
    max_episode_steps=1600,
    goal_max_dist= 10,
    goal_min_dist= 5,
    max_dist_boxes= 10,
    min_dist_boxes= 5,
    distance_threshold=1.2,
    relative_goals=False,
    num_prob_objects=1,
    prob_objects_prob=[0.5],
    num_random_walk_objects=1,
    num_magic_objects=0,
    num_dynamic_objects=0,
    physical_box=True,
    init_pos_noise=1.0
)

    env.reset()
    env.render()
    env.set_goal(np.asarray([5,0,5,0,0,5,0,5,0,5]), [[1,0]])
    mo_idx = env.model.body_name2id('pickupBox')
    # get x,y positions
    env.model.body_pos[mo_idx][:2] = [2,0]
    for _ in range(120):
        o, r, d, i = env.step([1, 0])
        print(i)
        env.render()
    input()
