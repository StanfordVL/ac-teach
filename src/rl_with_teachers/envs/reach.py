import gym
from gym import spaces
from gym.envs.registration import register
from gym.envs.robotics.fetch import reach, pick_and_place
from rl_with_teachers.teachers.reach import *
import numpy as np

class FetchOneGoalReachEnv(reach.FetchReachEnv):
    """
    A simple mujoco RL task to reach a point in 3D space with robot gripper.
    Meant as a sort of sanity check.
    """
    GOAL = np.array([1.25, 0.75, 0.5])
    def __init__(self):
        super().__init__('sparse')

        self.goal = FetchOneGoalReachEnv.GOAL

        self.start = np.array([1.34184371, 0.75, 0.5])

        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype='float32')
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

    def step(self, action):
        true_action = np.zeros(4)
        true_action[:2] = action

        # need to change action space since super class expects 4 dimensions
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)
        obs, reward, done, info = super().step(true_action)
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

        if reward==0.0:
            reward = 1
            done = True
        else:
            reward = 0.0
        return obs['observation'][:2], reward, done, info

    def reset(self):
        obs = super().reset()
        return obs['observation'][:2]

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self.sim.forward()

        gripper_offset = (np.random.random(3)-0.5)/2
        gripper_offset[2] = 0#gripper_offset[2]/3
        gripper_target = self.sim.data.get_site_xpos('robot0:grip') + gripper_offset
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        return True

    def _sample_goal(self):
        return FetchOneGoalReachEnv.GOAL.copy()

    def make_teachers(self, type, noise=None, drop=0, num_random=0):
        if type == 'optimal':
            return [OptimalReachAgent(self, self.goal, noise)]
        elif type == 'pick&place':
            halfway = HalfwayAgent(self.start, self.goal)
            switch = SwitchAgent(self.start, self.goal)
            return [halfway, switch]
        else:
            raise ValueError('Not a valid teacher type')

register(
        id='OneGoalReachEnv-v0',
        entry_point='rl_with_teachers.envs:FetchOneGoalReachEnv',
        max_episode_steps=50,
        )
