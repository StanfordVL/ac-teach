import gym
from gym.utils import seeding
from gym import spaces
from gym.envs.registration import register
from rl_with_teachers.teachers.path import *
import random
import numpy as np

DEFAULT_PATH_POINTS = np.array([[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]])
DEFAULT_PATH_POINTS_HARDER = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
CHECKPOINT_RADIUS = 0.05
class PointsPathEnv(gym.Env):
    """
    A simple Sanity check env for learning with experts.

    The env is D dimensional, with N path points.
    The goal is to move from the origin through each point in the path in sequence.
    The observation includes the current position and a binary indicator of whether each path point has been reached (so it is D+N dimensional).
    The actions are offsets with absolute value at most 0.045, so the default path is completable in 50 steps.
    """

    metadata = {
                'render.modes' : ['rgb_array'],
               }

    def __init__(self,
                 dims = 2,
                 points = DEFAULT_PATH_POINTS,
                 max_action_val = 0.045,
                 shuffle_order = True,
                 dense_reward = True,
                 goal_in_state = False,
                 render_q_quiver = False):
        self.dims = dims
        self.path_points = points
        self.shuffle_order = shuffle_order
        self.dense_reward = dense_reward
        self.points_ordering = list(range(len(points)))
        if goal_in_state:
            self.state_dims = 4
        else:
            self.state_dims = self.dims + len(points)
        if shuffle_order:
            self.state_dims+=1
        self.max_action_val = max_action_val
        self.reward_range = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dims,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1.0]*self.state_dims),
                                            high=np.array([1.0]*self.state_dims), dtype=np.float32)
        self.spec = False
        self.viewer = None
        self.agent = None
        self.goal_in_state = goal_in_state
        self.render_q_quiver = render_q_quiver
        self.drew_circles = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, -1., 1.)
        self.state[:self.dims] = self.state[:self.dims] + action * self.max_action_val

        reward = 0.0
        if (self.current_goal_idx < len(self.path_points) and
            np.linalg.norm(self.state[:self.dims] - self.current_goal) < CHECKPOINT_RADIUS):
                if self.dense_reward:
                    if self.current_goal_idx < len(self.points_ordering):
                        reward = (self.current_goal_idx+1.0)/(np.linalg.norm(self.state[:self.dims] - self.current_goal)+1.0) * 0.1
                else:
                    reward = 1.0
                self.current_goal_idx+=1
                if self.goal_in_state:
                    if self.current_goal_idx < len(self.points_ordering):
                        self.state[2:4] = self.current_goal
                    else:
                        self.state[2:] = 0.0
                else:
                    self.state[self.dims + self.points_ordering[self.current_goal_idx-1]] = 0.0
                    if self.current_goal_idx < len(self.points_ordering):
                        self.state[self.dims + self.points_ordering[self.current_goal_idx]] = 1.0

        if self.shuffle_order:
            self.state[-1] = len(self.path_points) - self.current_goal_idx

        done = self.current_goal_idx==len(self.path_points)

        return np.copy(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([0.0]*self.state_dims)
        self.current_goal_idx = 0
        if self.shuffle_order:
            random.shuffle(self.points_ordering)
            self.state[-1] = len(self.path_points)
        if self.goal_in_state:
            self.state[2:4] = self.current_goal
        else:
            self.state[self.dims + self.points_ordering[self.current_goal_idx]] = 1.0
        return np.copy(self.state)

    @property
    def current_goal(self):
        if self.current_goal_idx < len(self.points_ordering):
            return self.path_points[self.points_ordering[self.current_goal_idx]]
        else:
            return None

    def make_teachers(self, type, noise=None, drop=0, num_random=0):
        if type == 'optimal':
            return [OptimalPathAgent(self, noise, adversarial=False)]
        elif type == 'random':
            return [RandomAgent(self)]
        elif type == 'optimal_adversarial':
            return [OptimalPathAgent(self, noise, adversarial=False),
                    OptimalPathAgent(self, noise, adversarial=True)]
        elif type == 'optimal_halfway':
            # each part of path has midpoint teacher and switch teacher (start/goal)
            half_teacher = OptimalPathHalfwayAgent(self, noise=noise)
            switch_teacher = OptimalPathSwitchAgent(self, noise=noise)
            return [half_teacher, switch_teacher]
        elif type == 'axis':
            return [AxisAlignedPathAgent(self, a) for a in range(2)]
        elif type == 'axis_adversarial':
            teachers = [AxisAlignedPathAgent(self, a, adversarial=False) for a in range(2)]
            bad_teachers = [AxisAlignedPathAgent(self, a, adversarial=True) for a in range(2)]
            return teachers + bad_teachers
        elif type == 'one_goal':
            teachers = [OneGoalPathAgent(self, self.path_points[p], noise, adversarial=False) for p in range(len(self.path_points))]
            for i in range(drop):
                teachers.pop(random.randint(0, len(teachers)-1))
            for j in range(num_random):
                teachers.append(RandomAgent(self))
            return teachers
        elif type == 'one_goal_adversarial':
            teachers = [OneGoalPathAgent(self, self.path_points[p], noise, adversarial=False) for p in range(len(self.path_points))]
            bad_teachers = [OneGoalPathAgent(self, self.path_points[p], noise, adversarial=True) for p in range(len(self.path_points))]
            for i in range(drop):
                rand = random.randint(0, len(teachers)-1)
                teachers.pop(rand)
                bad_teachers.pop(rand)
            return teachers + bad_teachers
        elif type == 'halfway':
            # each part of path has midpoint teacher and switch teacher (start/goal)
            half_teachers = [OneGoalPathHalfwayAgent(self, path_id=p, noise=noise) for p in range(len(self.path_points))]
            switch_teachers = [OneGoalPathSwitchAgent(self, path_id=p, noise=noise) for p in range(len(self.path_points))]
            for i in range(drop):
                rand = random.randnint(0, len(half_teachers) - 1)
                half_teachers.pop(rand)
                switch_teachers.pop(rand)
            return half_teachers + switch_teachers
        else:
            raise ValueError('Not a valid teacher type')

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return
        SIZE = 500
        if self.viewer is None:
            self.viewer = gym_rendering.Viewer(SIZE, SIZE)

        x = SIZE/2.0 + self.state[0]*SIZE/2.0
        y = SIZE/2.0 + self.state[1]*SIZE/2.0
        polygon = gym_rendering.make_circle(radius=5, res=30, filled=True)
        state_tr = gym_rendering.Transform(translation=(x,y))
        polygon.add_attr(state_tr)
        self.viewer.add_onetime(polygon)

        #for i in range(len(self.path_points)):
        for i in range(self.current_goal_idx+1):
            if i >= len(self.path_points):
                break
            idx = self.points_ordering[i]
            point = SIZE/2.0 + self.path_points[idx]*SIZE/2.0
            tr = gym_rendering.Transform(translation=(point[0], point[1]))
            polygon = gym_rendering.make_circle(radius=CHECKPOINT_RADIUS/2.0*SIZE, res=30, filled=True)
            polygon.add_attr(tr)
            if i == self.current_goal_idx:
                polygon.set_color(1.0, 0., 0.)
            else:
                polygon.set_color(0.0, 1., 0.)
            self.viewer.add_onetime(polygon)

        if self.agent is not None and self.render_q_quiver:
            points = np.arange(-1, 1, 0.1)
            xx, yy = np.meshgrid(points, points, sparse=False)
            num_grid_points = xx.shape[0]*xx.shape[0]
            points = np.stack([xx,yy],-1).reshape([num_grid_points,2])
            goal_states = np.tile(self.state[self.dims:], [num_grid_points, 1])
            inputs = np.concatenate([points, goal_states], axis=-1)
            actions, q_vals = self.agent._policy(inputs, apply_noise=False)
            actions = actions.reshape([num_grid_points,2])
            q_max = np.max(q_vals)
            q_min = np.min(q_vals)
            action_mags = np.linalg.norm(actions,axis=1)
            mag_max = np.max(action_mags)
            mag_min = np.min(action_mags)
            for i in range(num_grid_points):
                start = SIZE/2.0 + SIZE*points[i]/2.0
                if not self.drew_circles:
                    tr = gym_rendering.Transform(translation=(start[0], start[1]))
                    c = gym_rendering.make_circle(radius=2, res=30, filled=True)
                    c.add_attr(tr)
                    self.viewer.add_geom(c)
                action = actions[i]
                action_mag = np.sum(np.abs(action))
                mag = action_mags[i]
                relative_mag = (mag-mag_min)/(mag_max-mag_min)
                action_normalized = action/np.sum(np.abs(action))*0.1*relative_mag
                end = start + SIZE/2.0*action_normalized
                q = q_vals[i,0]
                goodness = (q-q_min)/(q_max-q_min)
                color = (int((1-goodness)*255), 0, int(goodness*255))
                polygon = self.viewer.draw_line(start, end, color=color)
            self.drew_circles = True

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

register(
        id='PointsPath-v0',
        entry_point='rl_with_teachers.envs:PointsPathEnv',
        max_episode_steps=200,
        )

class SparsePointsPathEnv(PointsPathEnv):
    """
    Path env variant with sparse reward.
    """
    def __init__(self,
                 dims = 2,
                 points = DEFAULT_PATH_POINTS,
                 max_action_val = 0.045,
                 shuffle_order = True,
                 goal_in_state = True,
                 render_q_quiver = False):
        super().__init__(dims, points, max_action_val, shuffle_order, False, goal_in_state, render_q_quiver)

class SparseFixedOrderIndicatorPointsPathEnv(PointsPathEnv):
    """
    Path env variant with sparse reward and no shuffling of goal order.
    """
    def __init__(self,
                 dims = 2,
                 points = DEFAULT_PATH_POINTS,
                 max_action_val = 0.045,
                 goal_in_state = False,
                 render_q_quiver = False):
        super().__init__(dims, points, max_action_val, False, False, goal_in_state, render_q_quiver)

register(
        id='SparseFixedOrderIndicatorPointsPathEnv-v0',
        entry_point='rl_with_teachers.envs:SparseFixedOrderIndicatorPointsPathEnv',
        max_episode_steps=200,
        )

class SparseGoalInStatePointsPathEnv(SparsePointsPathEnv):
    """
    Path env variant with sparse reward and goal location in state.
    """
    def __init__(self,
                 dims = 2,
                 points = DEFAULT_PATH_POINTS,
                 max_action_val = 0.045,
                 shuffle_order = True,
                 render_q_quiver = False):
        super().__init__(dims, points, max_action_val, shuffle_order, True, render_q_quiver)

register(
        id='SparseGoalInStatePointsPath-v0',
        entry_point='rl_with_teachers.envs:SparseGoalInStatePointsPathEnv',
        max_episode_steps=200,
        )

class HarderSparseGoalInStatePointsPathEnv(SparsePointsPathEnv):
    """
    Path env variant with sparse reward and goal points spaces further apart.
    """

    def __init__(self,
                 dims = 2,
                 points = DEFAULT_PATH_POINTS_HARDER,
                 max_action_val = 0.045,
                 shuffle_order = True,
                 render_q_quiver = False):
        super().__init__(dims, points, max_action_val, False, shuffle_order, True, render_q_quiver)

register(
        id='HarderSparseGoalInStatePointsPath-v0',
        entry_point='rl_with_teachers.envs:HarderSparseGoalInStatePointsPathEnv',
        max_episode_steps=200,
        )
