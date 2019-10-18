from rl_with_teachers.teachers.base import TeacherPolicy
from rl_with_teachers.teachers.pick_place import PickAgent
from rl_with_teachers.teachers.pick_place import PlaceAgent
import numpy as np

class OriginalHookSweepAgent(TeacherPolicy):
    """
    For Fetch HookSweep environment. Based on RPL code.
    """
    def __init__(self, noise=None):
        from rpl_environments.envs.hook_controller import get_hook_control
        self.noise = noise
        self.actor = get_hook_control

    def __call__(self, obs):
        obs_dict = {'observation': np.array(obs[:-3]), 'desired_goal': np.array(obs[-3:])}
        action = self.actor(obs_dict)
        noisy_action = self.apply_noise(np.array(action))
        return noisy_action

class FullHookSweepPushAgent(TeacherPolicy):
    """
    For Fetch HookSweepPush environment. Decides which full agent to employ
    depending on where the goal position is.
    """
    def __init__(self, noise=None, short_hook=False, naive=False, cube_side_init=False):
        self.noise = noise
        self.short_hook = short_hook
        self.cube_side_init = cube_side_init
        self.full_sweeper = FullHookSweepAgent(naive=naive, cube_side_init=cube_side_init)
        self.full_pusher = FullHookPushAgent(short_hook=short_hook, naive=naive, cube_side_init=cube_side_init)

    def __call__(self, obs):
        # we know which task is being done based on the goal location
        goal = obs[-3:]
        if goal[0] > 1.8:
            action = self.full_pusher(obs)
        else:
            action = self.full_sweeper(obs)
        noisy_action = self.apply_noise(np.array(action))
        return noisy_action

    def reset(self):
        self.full_sweeper.reset()
        self.full_pusher.reset()

class FullHookSweepAgent(TeacherPolicy):
    """
    For Fetch HookSweep environment. A teacher that suboptimally solves the task.
    """
    def __init__(self, noise=None, naive=False, cube_side_init=False):
        self.noise = noise
        self.cube_side_init = cube_side_init
        self.picker = PickHookAgent()
        self.positioner = PositionForSweepHookAgent(cube_side_init=cube_side_init)
        if naive:
            self.sweeper = NaiveSweepHookAgent()
        else:
            self.sweeper = SweepHookAgent(cube_side_init=cube_side_init)

    def __call__(self, obs):
        if not self.picker.is_grasping(obs):
            action = self.picker(obs)
        elif not self.positioner.placed_close_enough(obs):
            action = self.positioner(obs)
        else:
            action = self.sweeper(obs)
        noisy_action = self.apply_noise(np.array(action))
        return noisy_action

    def reset(self):
        self.picker = PickHookAgent()
        self.positioner = PositionForSweepHookAgent()
        self.sweeper = SweepHookAgent()

class FullHookPushAgent(TeacherPolicy):
    """
    For Fetch HookPush environment. A teacher that suboptimally solves the task.
    """
    def __init__(self, noise=None, short_hook=False, naive=False, cube_side_init=False):
        self.noise = noise
        self.short_hook = short_hook
        self.cube_side_init = cube_side_init
        self.picker = PickHookAgent()
        self.positioner = PositionForPushHookAgent(short_hook=short_hook, cube_side_init=cube_side_init)
        if naive:
            self.pusher = NaivePushHookAgent(short_hook=short_hook)
        else:
            self.pusher = PushHookAgent(short_hook=short_hook, cube_side_init=cube_side_init)

    def __call__(self, obs):
        if not self.picker.is_grasping(obs):
            action = self.picker(obs)
        elif not self.positioner.placed_close_enough(obs):
            action = self.positioner(obs)
        else:
            action = self.pusher(obs)
        noisy_action = self.apply_noise(np.array(action))
        return noisy_action

    def reset(self):
        self.picker.reset()
        self.positioner.reset()
        self.pusher.reset()

class PickHookAgent(PickAgent):
    """
    For Fetch HookSweep environment with single cube position and goal.


    It first moves translationally into place before moving vertically towards
    the target. Once it is sufficiently close, the agent attempts
    to grab the target object.
    """
    def __call__(self, obs, obj_index=25):
        return super().__call__(obs, obj_index=obj_index)

    def is_close_to_grasping(self, obs, obj_index=25):
        return super().is_close_to_grasping(obs, obj_index=obj_index)

    def is_grasping(self, obs, obj_index=25):
        return super().is_grasping(obs, obj_index=obj_index)

class PositionHookAgent(PlaceAgent):
    """
    For Fetch HookSweep environment with single cube position and goal.

    This agent assumes that it already has the target grasped and just follows a
    simple reaching policy to move the target to the goal.
    """
    def __call__(self, obs, obj_index=25):
        return super().__call__(obs, obj_index=obj_index)

class PositionForSweepPushHookAgent(TeacherPolicy):
    """
    For Fetch HookSweepPush environment. Positions for sweeping or pushing
    depending on the goal position.
    """
    def __init__(self, noise=None, short_hook=False, cube_side_init=False):
        self.noise  = noise
        self.short_hook = short_hook
        self.cube_side_init = cube_side_init
        self.sweep_position_hooker = PositionForSweepHookAgent(cube_side_init=cube_side_init)
        self.push_position_hooker = PositionForPushHookAgent(short_hook=short_hook, cube_side_init=cube_side_init)

    def __call__(self, obs):
        # we know which task is being done based on the goal location
        goal = obs[-3:]
        if goal[0] > 1.8:
            action = self.push_position_hooker(obs)
        else:
            action = self.sweep_position_hooker(obs)
        noisy_action = self.apply_noise(np.array(action))
        return noisy_action

    def reset(self):
        self.sweep_position_hooker.reset()
        self.push_position_hooker.reset()

class PositionForSweepHookAgent(PositionHookAgent):
    """
    For Fetch HookSweep environment with single cube position and goal.

    Grabs goal from observation based on where target object to sweep is.
    It takes an axis-aligned path to the goal.
    """
    def __init__(self, noise=None, cube_side_init=False):
        self.cube_side_init = cube_side_init
        axis_aligned = True
        if self.cube_side_init:
            axis_aligned = False
        super().__init__(goal=None, noise=noise, axis_aligned=axis_aligned, use_gripper=False, place_parabolic=False)

    def placed_close_enough(self, obs):
        gripper_pos = obs[:3]
        obj_pos = obs[3:6]
        hook_pos = obs[25:28]
        target_hook_pos = np.array([obj_pos[0] - 0.5, obj_pos[1] - 0.05, 0.45])
        self.goal = target_hook_pos

        # Some tolerance to stop positioning in cases where we don't have to.
        delta = obj_pos[:2] - gripper_pos[:2]
        delta /= np.linalg.norm(delta)
        angle = (180. / np.pi) * np.arccos(delta[0])

        return super().placed_close_enough(obs) or angle < 15.

    def __call__(self, obs):
        obj_pos = obs[3:6]
        hook_pos = obs[25:28]
        target_hook_pos = np.array([obj_pos[0] - 0.5, obj_pos[1] - 0.05, 0.45])

        self.goal = target_hook_pos
        return super().__call__(obs)


class PositionForPushHookAgent(PositionHookAgent):
    """
    For Fetch HookPush environment with single cube position and goal.

    Grabs goal from observation based on where target object to push is.
    """
    def __init__(self, noise=None, short_hook=False, cube_side_init=False):
        self.short_hook = short_hook
        self.cube_side_init = cube_side_init
        axis_aligned = True
        if self.cube_side_init:
            axis_aligned = False
        super().__init__(goal=None, noise=noise, axis_aligned=axis_aligned, use_gripper=False, place_parabolic=False)

    def placed_close_enough(self, obs):
        obj_pos = obs[3:6]
        hook_pos = obs[25:28]
        if self.cube_side_init:
            # target_hook_pos = np.array([obj_pos[0] - 0.8, obj_pos[1] + 0.1, 0.45])
            target_hook_pos = np.array([obj_pos[0] - 0.8, obj_pos[1] - 0.09, 0.45])
        elif self.short_hook:
            target_hook_pos = np.array([obj_pos[0] - 0.6, obj_pos[1] - 0.09, 0.45])
        else:
            target_hook_pos = np.array([obj_pos[0] - 0.7, obj_pos[1] - 0.09, 0.45])
        self.goal = target_hook_pos

        gripper_pos = obs[:3]
        delta = obj_pos[:2] - gripper_pos[:2]
        delta /= np.linalg.norm(delta)
        angle = (180. / np.pi) * np.arccos(delta[0])
        if self.cube_side_init:
            return super().placed_close_enough(obs) or angle < 10.
        return super().placed_close_enough(obs)  or angle < 15.

    def __call__(self, obs):
        obj_pos = obs[3:6]
        hook_pos = obs[25:28]
        if self.cube_side_init:
            # target_hook_pos = np.array([obj_pos[0] - 0.8, obj_pos[1] + 0.1, 0.45])
            target_hook_pos = np.array([obj_pos[0] - 0.8, obj_pos[1] - 0.09, 0.45])
        elif self.short_hook:
            target_hook_pos = np.array([obj_pos[0] - 0.6, obj_pos[1] - 0.09, 0.45])
        else:
            target_hook_pos = np.array([obj_pos[0] - 0.7, obj_pos[1] - 0.09, 0.45])
        # target_hook_pos = np.array([obj_pos[0] - 0.5, obj_pos[1], 0.45])

        if self.cube_side_init:
            action = 0.5 * np.clip((target_hook_pos - hook_pos) / 0.05, -1., 1.)
            # action = 10. * (target_hook_pos - hook_pos)
            action = np.array([action[0], action[1], action[2], -1.])
            # action = np.clip(action, -1., 1.)
            return self.apply_noise(np.array(action))

        self.goal = target_hook_pos
        return super().__call__(obs)

class SweepHookAgent(TeacherPolicy):
    """
    For Fetch HookSweep environment with single cube position and goal.

    Assumes that the hook has been grabbed, and just sweeps backward to goal location.
    It does not handle any alignment, but rather just reaches in the x and y direction
    to a hook position behind the object goal.

    Alternative: sweeps in a direction from current object location to goal.
    """
    def __init__(self, noise=None, cube_side_init=False):
        self.noise = noise
        self.cube_side_init = cube_side_init

    def __call__(self, obs):
        loc = obs[:3]
        obj_pos = obs[3:6]
        obj_goal = obs[-3:]

        target_hook_pos = np.array(obj_goal)
        target_hook_pos[0] -= 0.6
        target_hook_pos[1] -= 0.05
        target_hook_pos[2] = 0.45
        delta_x = target_hook_pos - loc

        action = 0.5 * np.clip(delta_x / 0.05, -1., 1.)

        if self.cube_side_init:
            delta_x = obj_goal - obj_pos
            delta_x[:2] = delta_x[:2] / np.linalg.norm(delta_x[:2])
            delta_x[2] = 0.
            # action = [0.4 * delta_x[0], 0.4 * delta_x[1], 0.]
            action = np.clip(delta_x, -1., 1.)

        action = np.array([action[0], action[1], action[2], -1.])
        return self.apply_noise(action)

class PushHookAgent(TeacherPolicy):
    """
    For Fetch HookPush environment with single cube position and goal.

    Assumes that the hook has been grabbed, and just pushes forward to goal location.
    It does not handle any alignment, but rather just reaches in the x and y direction
    to a hook position behind the object goal.

    Alternative: sweeps in a direction from current object location to goal.
    """
    def __init__(self, noise=None, short_hook=False, cube_side_init=False):
        self.noise = noise
        self.short_hook = short_hook
        self.cube_side_init = cube_side_init

    def __call__(self, obs):
        loc = obs[:3]
        obj_pos = obs[3:6]
        obj_goal = obs[-3:]

        target_hook_pos = np.array(obj_goal)
        if self.short_hook:
            target_hook_pos[0] += 0.1
        target_hook_pos[0] -= 0.69
        target_hook_pos[1] -= 0.1
        target_hook_pos[2] = 0.45
        delta_x = target_hook_pos - loc
        action = 0.5 * np.clip(delta_x / 0.05, -1., 1.)

        if self.cube_side_init:
            delta_x = obj_goal - obj_pos
            delta_x[:2] = delta_x[:2] / np.linalg.norm(delta_x[:2])
            delta_x[2] = 0.
            action = [0.4 * delta_x[0], 0.4 * delta_x[1], 0.]
            # action = np.clip(delta_x, -1., 1.)

        action = np.array([action[0], action[1], action[2], -1.])
        return self.apply_noise(action)

class NaiveSweepHookAgent(TeacherPolicy):
    """
    For Fetch HookSweep environment with single cube position and goal.

    Just moves end effector in negative x direction. Assumes that the hook
    has been grabbed (so it keeps actuating the gripper to be closed).
    """
    def __init__(self, noise=None):
        self.noise = noise

    def __call__(self, obs):
        # make the gripper float a little above the table to get good leverage for the hook
        gripper_pos = obs[:3]
        delta_z = 0.45 - gripper_pos[2]
        delta_z = 0.5 * np.clip(delta_z / 0.05, -1., 1.)

        action = np.array([-0.5, 0., delta_z, -1.])
        return self.apply_noise(action)

class NaivePushHookAgent(TeacherPolicy):
    """
    For Fetch HookPush environment with single cube position and goal.

    Just moves end effector in positive x direction. Assumes that the hook
    has been grabbed (so it keeps actuating the gripper to be closed).
    """
    def __init__(self, noise=None, short_hook=False):
        self.noise = noise

    def __call__(self, obs):
        # make the gripper float a little above the table to get good leverage for the hook
        gripper_pos = obs[:3]
        delta_z = 0.45 - gripper_pos[2]
        delta_z = 0.5 * np.clip(delta_z / 0.05, -1., 1.)

        action = np.array([0.5, 0., delta_z, -1.])
        return self.apply_noise(action)
