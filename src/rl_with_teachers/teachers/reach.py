from rl_with_teachers.teachers.base import TeacherPolicy
import numpy as np

class HalfwayAgent(TeacherPolicy):
    """
    For Fetch reach environment with single start and goal. This agent
    always reaches for the midpoint between the start and the goal.
    """
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.halfway = (start + goal) / 2.
        self.offset = goal - start

    def __call__(self, obs):
        action = np.array([0., 0., 0., 1.])
        action[:3] = (self.halfway + self.offset * 0.05 - obs[0:3]) * 3.0
        return action

class SwitchAgent(TeacherPolicy):
    """
    For Fetch reach environment with single start and goal. This agent
    reaches for the start position or the goal position, depending on
    which is closer.
    """
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.offset = goal - start

    def __call__(self, obs):
        action = np.array([0., 0., 0., 1.])

        pos = obs[:3]
        if np.linalg.norm(pos - self.goal) < np.linalg.norm(pos - self.start):
            goal_now = self.goal - self.offset * 0.15 # 85% of the way to the true goal
        else:
            goal_now = self.start + self.offset * 0.05 # 5% of the way from the start

        action[:3] = (goal_now - obs[0:3]) * 3.0
        return action

class OptimalReachAgent(TeacherPolicy):
    """
    For Fetch reach environment with single start and goal. This agent
    reaches for the goal position unconditionally.
    """
    def __init__(self, env, goal, noise=None):
        self.env = env
        self.goal = goal
        self.noise = noise

    def __call__(self, obs):
        loc = obs[0:2]
        action = np.clip((self.goal[0:2] - loc)/0.05, -1.0 , 1.0)
        action = action[:2] # no z-action for now
        return self.apply_noise(action)
