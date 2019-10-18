from rl_with_teachers.teachers.base import TeacherPolicy

class OptimalPathAgent(TeacherPolicy):
    """
    A path agent that just goes straight to current goal at maximum seed.
    """
    def __init__(self, env, noise=None, adversarial=False):
        self.env = env
        self.noise = noise
        self.adversarial = adversarial

    def __call__(self, obs):
        loc = obs[:self.env.dims]
        goal = self.env.current_goal
        if self.env.current_goal is not None:
            action = np.clip((self.env.current_goal - loc) / self.env.max_action_val, -1.0 , 1.0)
            if self.adversarial:
                action *= -1. # move away from target instead of towards target
        else:
            action = self.env.action_space.sample()
        return self.apply_noise(action)

class OptimalPathHalfwayAgent(OptimalPathAgent):
    """
    For Path environment.
    This agent always reaches for the midpoint between the start and the current goal.
    """
    def __init__(self, env, noise=None):
        self.env = env
        self.noise = noise
        self.adversarial = False

    def __call__(self, obs):
        if self.env.current_goal_idx >= len(self.env.points_ordering):
            path_id = len(self.env.points_ordering) - 1
        else:
            path_id = self.env.current_goal_idx

        if path_id == 0:
            start = np.zeros(self.env.dims)
        else:
            start = self.env.path_points[self.env.points_ordering[path_id - 1]]
        goal = self.env.path_points[self.env.points_ordering[path_id]]
        halfway = (start + goal) / 2.
        offset = goal - start

        loc = obs[:self.env.dims]
        action = np.clip((halfway - loc) / self.env.max_action_val, -1.0 , 1.0)

        # action = np.array([0., 0., 0., 1.])
        # action[:3] = (self.halfway + self.offset * 0.05 - obs[0:3]) * 3.0
        return self.apply_noise(action)

class OptimalPathSwitchAgent(OptimalPathAgent):
    """
    For Path environment.
    This agent reaches for the start position or the current goal position, depending on
    which is closer.
    """
    def __init__(self, env, noise=None):
        self.env = env
        self.noise = noise
        self.adversarial = False

    def __call__(self, obs):
        if self.env.current_goal_idx >= len(self.env.points_ordering):
            path_id = len(self.env.points_ordering) - 1
        else:
            path_id = self.env.current_goal_idx

        if path_id == 0:
            start = np.zeros(self.env.dims)
        else:
            start = self.env.path_points[self.env.points_ordering[path_id - 1]]
        goal = self.env.path_points[self.env.points_ordering[path_id]]
        offset = goal - start

        loc = obs[:self.env.dims]
        if np.linalg.norm(loc - goal) < np.linalg.norm(loc - start):
            to_reach = goal
            # goal_now = self.goal - self.offset * 0.15 # 85% of the way to the true goal
        else:
            to_reach = start
            # goal_now = self.start + self.offset * 0.05 # 5% of the way from the start

        action = np.clip((to_reach - loc) / self.env.max_action_val, -1.0 , 1.0)
        return self.apply_noise(action)

class AxisAlignedPathAgent(OptimalPathAgent):
    """
    For Path environment.
    A suboptimal agent that only moves along one axis towards goal (and so never gets to it,
    if any other axis requires movement).
    """
    def __init__(self, env, axis, noise=None, adversarial=False):
        self.env = env
        self.axis = axis
        self.noise = noise
        self.adversarial = adversarial

    def __call__(self, obs):
        action = super().__call__(obs)
        for i in range(len(action)):
            if i!=self.axis:
                action[i]=0
        return self.apply_noise(action)

class OneGoalPathAgent(OptimalPathAgent):
    """
    For Path environment.
    A suboptimal agent that only gets to one location in env, and then does nothing.
    """
    def __init__(self, env, goal, noise=None, adversarial=False):
        self.env = env
        self.goal = goal
        self.noise = noise
        self.adversarial = adversarial

    def is_callable(self, obs):
        loc = obs[:self.env.dims]
        return np.sum(np.abs(loc-self.goal)) > 0.01

    def __call__(self, obs):
        loc = obs[:self.env.dims]
        action = np.clip((self.goal - loc) / self.env.max_action_val, -1.0 , 1.0)
        if self.adversarial:
            action *= -1. # move away from target instead of towards target
        return self.apply_noise(action)

class OneGoalPathHalfwayAgent(OptimalPathAgent):
    """
    For Path environment with single start and goal (subtask). This agent
    always reaches for the midpoint between the start and the goal.
    """
    def __init__(self, env, path_id, noise=None):
        self.env = env
        self.path_id = path_id # which section of path this teacher is responsible for
        self.noise = noise
        self.adversarial = False

    def __call__(self, obs):
        if self.path_id == 0:
            start = np.zeros(self.env.dims)
        else:
            start = self.env.path_points[self.env.points_ordering[self.path_id - 1]]
        goal = self.env.path_points[self.env.points_ordering[self.path_id]]
        halfway = (start + goal) / 2.
        offset = goal - start

        loc = obs[:self.env.dims]
        action = np.clip((halfway - loc) / self.env.max_action_val, -1.0 , 1.0)

        # action = np.array([0., 0., 0., 1.])
        # action[:3] = (self.halfway + self.offset * 0.05 - obs[0:3]) * 3.0
        return self.apply_noise(action)

class OneGoalPathSwitchAgent(OptimalPathAgent):
    """
    For Path environment with single start and goal (subtask). This agent
    reaches for the start position or the goal position, depending on
    which is closer.
    """
    def __init__(self, env, path_id, noise=None):
        self.env = env
        self.path_id = path_id
        self.noise = noise
        self.adversarial = False

    def __call__(self, obs):
        if self.path_id == 0:
            start = np.zeros(self.env.dims)
        else:
            start = self.env.path_points[self.env.points_ordering[self.path_id - 1]]
        goal = self.env.path_points[self.env.points_ordering[self.path_id]]
        offset = goal - start

        loc = obs[:self.env.dims]
        if np.linalg.norm(loc - goal) < np.linalg.norm(loc - start):
            to_reach = goal
            # goal_now = self.goal - self.offset * 0.15 # 85% of the way to the true goal
        else:
            to_reach = start
            # goal_now = self.start + self.offset * 0.05 # 5% of the way from the start

        action = np.clip((to_reach - loc) / self.env.max_action_val, -1.0 , 1.0)
        return self.apply_noise(action)
