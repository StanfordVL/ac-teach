import os
import gym
from gym import spaces
from gym.envs.registration import register
from gym.envs.robotics import rotations, fetch_env
from gym.envs.robotics import utils as gym_robotics_utils
from rl_with_teachers.teachers.hook_sweep import *
import numpy as np

### Residual Policy Learning Fetch Variants ###
class FetchHookSweepEnv(fetch_env.FetchEnv):
    """
    Mujoco env for robotics sweeping and pushing with a hook object.
    Configurable for multible variants of difficulty.
    """
    def __init__(
        self,
        xml_file=None,
        sparse=True,
        better_dense_reward=False,
        easy_init=False,
        easy_task=False,
        easier_task=False,
        grasp_reward=True,
        push_task=False,
        short_hook=False,
        close_push_init=False,
        cube_side_init=False,
        randomize_cube=False,
    ):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.35, 0.35, 0.4, 1., 0., 0., 0.],
        }

        if xml_file is None:
            #Go 3 folders up to base of rl_with_teachers dir
            package_path = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))))
            if short_hook:
                assert(push_task)
                xml_file = os.path.join(package_path, 'assets', 'sweep_short.xml')
            else:
                xml_file = os.path.join(package_path, 'assets', 'sweep.xml')

        # base goal position, from which all other are derived
        self._base_goal_pos = np.array([1.65, 0.75, 0.42])

        self.is_sparse = sparse
        self.better_dense_reward = better_dense_reward
        self.easy_init = easy_init
        self.easy_task = easy_task
        self.easier_task = easier_task
        self.grasp_reward = grasp_reward
        self.push_task = push_task
        self.short_hook = short_hook
        self.close_push_init = close_push_init
        self.cube_side_init = cube_side_init
        self.randomize_cube = randomize_cube

        # derive goal position from settings
        self._goal_pos = np.array(self._base_goal_pos)
        if self.push_task:
            self._goal_pos[0] += 0.40
            self._goal_pos[1] -= 0.31
            if self.cube_side_init:
                self._goal_pos[1] += 0.1
            if self.short_hook:
                self._goal_pos[0] -= 0.07
        elif self.easier_task:
            self._goal_pos[0] -= 0.3
            self._goal_pos[1] -= 0.3
        elif self.easy_task:
            self._goal_pos[1] -= 0.15

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse' if sparse else 'dense')

        obs = self._get_obs()
        total_shape = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(total_shape,), dtype='float32')

        # for detecting contact between hook and gripper
        self.hook_geom_id = self.sim.model.geom_name2id("hook_base")
        self.r_finger_geom = self.sim.model.geom_name2id("robot0:r_gripper_finger_link")
        self.l_finger_geom = self.sim.model.geom_name2id("robot0:l_gripper_finger_link")

    def render(self, mode="human", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            width, height = 3350, 1800
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

        return super(FetchHookSweepEnv, self).render(*args, **kwargs)

    def _sample_goal(self):
        goal_pos = self._goal_pos.copy()
        if not self.easy_task and not self.easier_task:
            goal_pos[:2] += self.np_random.uniform(-0.05, 0.05)
        return goal_pos

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -24.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # object_xpos_x = 1.65 + self.np_random.uniform(-0.05, 0.05)
        while True:
            if self.cube_side_init:
                object_xpos_x = 1.75
                object_xpos_y = 0.6
            elif self.push_task:
                if self.close_push_init:
                    object_xpos_x = 1.8
                else:
                    object_xpos_x = 1.9
                if self.short_hook:
                    object_xpos_x -= 0.1
                object_xpos_y = 0.44
            elif self.easier_task:
                object_xpos_x = 1.7
                object_xpos_y = 0.45
            elif self.easy_task:
                object_xpos_x = 1.8
                object_xpos_y = 0.6
            else:
                object_xpos_x = 1.8
                object_xpos_y = 0.75
            if self.randomize_cube:
                object_xpos_x += self.np_random.uniform(-0.025, 0.025)
                object_xpos_y += self.np_random.uniform(-0.025, 0.025)
            if (object_xpos_x - self._goal_pos[0])**2 + (object_xpos_y - self._goal_pos[1])**2 >= 0.01:
                break
        self._object_xpos = np.array([object_xpos_x, object_xpos_y])

        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = self._object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # ### TODO: define reward... ###
        # assert(self.is_sparse)

        if self.is_sparse:
            if reward == 0.:
                reward = 1.
            else:
                reward = 0.
        elif self.better_dense_reward:
            reward = 0.

            grip_pos = obs['observation'][:3]
            obj_pos = obs['observation'][3:6]
            hook_pos = obs['observation'][25:28]
            obj_goal = obs['desired_goal']

            # hook_dist = np.linalg.norm(grip_pos - hook_pos)
            # hook_reward = 1. - np.tanh(10.0 * hook_dist)

            ### give sparse reward for grasping hook ###
            touch_left_finger = False
            touch_right_finger = False
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 == self.l_finger_geom and c.geom2 == self.hook_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.hook_geom_id and c.geom2 == self.l_finger_geom:
                    touch_left_finger = True
                if c.geom1 == self.r_finger_geom and c.geom2 == self.hook_geom_id:
                    touch_right_finger = True
                if c.geom1 == self.hook_geom_id and c.geom2 == self.r_finger_geom:
                    touch_right_finger = True
            if (touch_left_finger and touch_right_finger) and self.grasp_reward:
                reward += 0.25

            ### give dense reward for distance from block to goal
            goal_dist = np.linalg.norm(obj_goal - obj_pos)
            goal_reward = 1. - np.tanh(10.0 * goal_dist)
            # relative_dist = goal_dist / self.initial_distance
            # goal_reward = 1. - np.tanh(5.0 * relative_dist)
            # goal_reward = 1. - np.tanh(10.0 * goal_dist)
            reward += goal_reward

        # make sure to include goal in observation if we are changing the goal
        obs_to_ret = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs_to_ret, reward, done, info

    def reset(self):

        # derive goal position from settings
        self._goal_pos = np.array(self._base_goal_pos)
        if self.push_task:
            self._goal_pos[0] += 0.40
            self._goal_pos[1] -= 0.31
            if self.cube_side_init:
                self._goal_pos[1] += 0.1
            if self.short_hook:
                self._goal_pos[0] -= 0.07
        elif self.easier_task:
            self._goal_pos[0] -= 0.3
            self._goal_pos[1] -= 0.3
        elif self.easy_task:
            self._goal_pos[1] -= 0.15

        obs = super().reset()
        # make sure to include goal in observation if we are changing the goal
        obs_to_ret = np.concatenate([obs['observation'], obs['desired_goal']])

        # record initial distance from object to goal
        self.initial_distance = np.linalg.norm(obs_to_ret[3:6] - obs_to_ret[-3:])
        return obs_to_ret

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])

        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

    def _env_setup(self, initial_qpos):
        """
        TODO: We can initialize the end effector using this method.
        """
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        gym_robotics_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        if self.easy_init:
            ### new position, closer to hook at start ###
            gripper_target = np.array([-0.7, -0.4, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        else:
            ### the old, initial position ###
            gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')

        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def make_teachers(self, type, noise=None, drop=0, num_random=0):
        if type == 'optimal':
            return [FullHookSweepAgent(noise=noise, cube_side_init=self.cube_side_init)]
        elif type == 'optimal_naive':
            return [FullHookSweepAgent(noise=noise, naive=True, cube_side_init=self.cube_side_init)]
        elif type == 'partial':
            pick_hooker = PickHookAgent(noise=noise)
            place_hooker = PositionForSweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            sweep_hooker = SweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            teachers = [pick_hooker, place_hooker, sweep_hooker]
            for i in range(num_random):
                teachers.append(RandomAgent(self))
            return teachers
        elif type == 'partial_naive':
            pick_hooker = PickHookAgent(noise=noise)
            place_hooker = PositionForSweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            sweep_hooker = NaiveSweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            teachers = [pick_hooker, place_hooker, sweep_hooker]
            for i in range(num_random):
                teachers.append(RandomAgent(self))
            return teachers
        elif type == 'original':
            return [OriginalHookSweepAgent(noise=noise)]
        else:
            raise ValueError('Not a valid teacher type '+type)

register(
        id='FetchHookSweep-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepEnv',
        max_episode_steps=100,
        )

class FetchHookSweepSparseEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=True)

register(
        id='FetchHookSweepSparse-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepSparseEnv',
        max_episode_steps=100,
        )

class FetchHookSweepDenseEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False)

register(
        id='FetchHookSweepDense-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepDenseEnv',
        max_episode_steps=100,
        )

class FetchHookSweepBetterDenseEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True)

register(
        id='FetchHookSweepBetterDense-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepBetterDenseEnv',
        max_episode_steps=100,
        )

## grasping rewards ##
class FetchHookSweepBetterDenseEasyInitEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easy_init=True)

register(
        id='FetchHookSweepBetterDenseEasyInit-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepBetterDenseEasyInitEnv',
        max_episode_steps=100,
        )

class FetchHookSweepBetterDenseEasyTaskEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easy_init=True, easy_task=True)

register(
        id='FetchHookSweepBetterDenseEasyTask-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepBetterDenseEasyTaskEnv',
        max_episode_steps=100,
        )

class FetchHookSweepBetterDenseEasierTaskEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easy_init=True, easier_task=True)

register(
        id='FetchHookSweepBetterDenseEasierTask-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepBetterDenseEasierTaskEnv',
        max_episode_steps=100,
        )

## no grasping rewards ##
class FetchHookSweepBetterDenseEasyInitNoGraspEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easy_init=True, grasp_reward=False)

register(
        id='FetchHookSweepBetterDenseEasyInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepBetterDenseEasyInitNoGraspEnv',
        max_episode_steps=100,
        )

class FetchHookSweepBetterDenseEasyTaskNoGraspEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easy_init=True, easy_task=True, grasp_reward=False)

register(
        id='FetchHookSweepBetterDenseEasyTaskNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepBetterDenseEasyTaskNoGraspEnv',
        max_episode_steps=100,
        )

class FetchHookSweepBetterDenseEasierTaskNoGraspEnv(FetchHookSweepEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easy_init=True, easier_task=True, grasp_reward=False)

register(
        id='FetchHookSweepBetterDenseEasierTaskNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepBetterDenseEasierTaskNoGraspEnv',
        max_episode_steps=100,
        )

class FetchHookPushEnv(FetchHookSweepEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_teachers(self, type, noise=None, drop=0, num_random=0):
        if type == 'optimal':
            return [FullHookPushAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)]
        elif type == 'optimal_naive':
            return [FullHookPushAgent(noise=noise, short_hook=self.short_hook, naive=True, cube_side_init=self.cube_side_init)]
        elif type == 'partial':
            pick_hooker = PickHookAgent(noise=noise)
            place_hooker = PositionForPushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            push_hooker = PushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            teachers = [pick_hooker, place_hooker, push_hooker]
            for i in range(num_random):
                teachers.append(RandomAgent(self))
            return teachers
        elif type == 'partial_naive':
            pick_hooker = PickHookAgent(noise=noise)
            place_hooker = PositionForPushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            push_hooker = NaivePushHookAgent(noise=noise, short_hook=self.short_hook)
            teachers = [pick_hooker, place_hooker, push_hooker]
            for i in range(num_random):
                teachers.append(RandomAgent(self))
            return teachers
        else:
            raise ValueError('Not a valid teacher type '+type)

## grasping rewards ##
class FetchHookPushBetterDenseEasyInitEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, grasp_reward=True, easy_init=True)

register(
        id='FetchHookPushBetterDenseEasyInit-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseEasyInitEnv',
        max_episode_steps=100,
        )


class FetchHookPushBetterDenseCloseInitEasyInitEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, close_push_init=True, grasp_reward=True, easy_init=True)

register(
        id='FetchHookPushBetterDenseCloseInitEasyInit-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseCloseInitEasyInitEnv',
        max_episode_steps=100,
        )

# close init, but no easy init
class FetchHookPushBetterDenseCloseInitEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, close_push_init=True, grasp_reward=True, easy_init=False)

register(
        id='FetchHookPushBetterDenseCloseInit-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseCloseInitEnv',
        max_episode_steps=100,
        )

## no grasping rewards ##
class FetchHookPushBetterDenseEasyInitNoGraspEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookPushBetterDenseEasyInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseEasyInitNoGraspEnv',
        max_episode_steps=100,
        )


class FetchHookPushBetterDenseCloseInitEasyInitNoGraspEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, close_push_init=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookPushBetterDenseCloseInitEasyInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseCloseInitEasyInitNoGraspEnv',
        max_episode_steps=100,
        )

# close init, but no easy init
class FetchHookPushBetterDenseCloseInitNoGraspEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, close_push_init=True, grasp_reward=False, easy_init=False)

register(
        id='FetchHookPushBetterDenseCloseInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseCloseInitNoGraspEnv',
        max_episode_steps=100,
        )


# Short Hook Environments, unused for now...
class FetchHookPushBetterDenseShortHookNoGraspEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, short_hook=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookPushBetterDenseShortHookNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseShortHookNoGraspEnv',
        max_episode_steps=100,
        )

class FetchHookPushBetterDenseShortHookCloseInitNoGraspEnv(FetchHookPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, push_task=True, short_hook=True, close_push_init=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookPushBetterDenseShortHookCloseInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookPushBetterDenseShortHookCloseInitNoGraspEnv',
        max_episode_steps=100,
        )


class FetchHookSweepPushEnv(FetchHookSweepEnv):
    """
    Combines both sweeping and pushing into one task, where depending on the goal
    initialization, sweeping or pushing must happen.
    """
    def __init__(self, push_prob=0.5, **kwargs):
        # probability of choosing pushing task (over sweeping) on every reset
        self.push_prob = push_prob
        super().__init__(**kwargs)

    def reset(self):
        # randomly determine whether to do push task or not
        self.push_task = False
        if np.random.rand() < self.push_prob:
            self.push_task = True
        return super().reset()

    def make_teachers(self, type, noise=None, drop=0, num_random=0):
        if type == 'optimal':
            return [FullHookSweepPushAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)]
        elif type == 'optimal_naive':
            return [FullHookSweepPushAgent(noise=noise, short_hook=self.short_hook, naive=True, cube_side_init=self.cube_side_init)]
        elif type == 'optimal_partial':
            # special case, where must choose between two optimal teachers based on task
            sweeper = FullHookSweepAgent(noise=noise, cube_side_init=self.cube_side_init)
            pusher = FullHookPushAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            teachers = [sweeper, pusher]
        elif type == 'optimal_partial_naive':
            # special case, where must choose between two optimal teachers based on task
            sweeper = FullHookSweepAgent(noise=noise, naive=True, cube_side_init=self.cube_side_init)
            pusher = FullHookPushAgent(noise=noise, short_hook=self.short_hook, naive=True, cube_side_init=self.cube_side_init)
            teachers = [sweeper, pusher]
            for i in range(num_random):
                teachers.append(RandomAgent(self))
            return teachers
        elif type == 'partial':
            pick_hooker = PickHookAgent(noise=noise)
            position_hooker = PositionForSweepPushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            sweep_hooker = SweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            push_hooker = PushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            teachers = [pick_hooker, position_hooker, sweep_hooker, push_hooker]
        elif type == 'partial_no_push':
            pick_hooker = PickHookAgent(noise=noise)
            position_hooker = PositionForSweepPushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            sweep_hooker = SweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            push_hooker = PushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            teachers = [pick_hooker, position_hooker, sweep_hooker]
        elif type == 'partial_no_sweep':
            pick_hooker = PickHookAgent(noise=noise)
            position_hooker = PositionForSweepPushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            sweep_hooker = SweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            push_hooker = PushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            teachers = [pick_hooker, position_hooker, push_hooker]
        elif type == 'partial_no_push':
            pick_hooker = PickHookAgent(noise=noise)
            position_hooker = PositionForSweepPushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            sweep_hooker = SweepHookAgent(noise=noise, cube_side_init=self.cube_side_init)
            push_hooker = PushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            teachers = [pick_hooker, position_hooker, sweep_hooker]
        elif type == 'partial_naive':
            pick_hooker = PickHookAgent(noise=noise)
        elif type == 'partial_naive':
            pick_hooker = PickHookAgent(noise=noise)
        elif type == 'partial_naive':
            pick_hooker = PickHookAgent(noise=noise)
            position_hooker = PositionForSweepPushHookAgent(noise=noise, short_hook=self.short_hook, cube_side_init=self.cube_side_init)
            sweep_hooker = NaiveSweepHookAgent(noise=noise)
            push_hooker = NaivePushHookAgent(noise=noise, short_hook=self.short_hook)
            teachers = [pick_hooker, position_hooker, sweep_hooker, push_hooker]
        else:
            raise ValueError('Not a valid teacher type '+type)
        for i in range(num_random):
            teachers.append(RandomAgent(self))
        return teachers

class FetchHookSweepPushDenseEasyInitEasierTaskCloseInitEnv(FetchHookSweepPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=False, easier_task=True, close_push_init=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookSweepPushDenseEasyInitEasierTaskCloseInit-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepPushDenseEasyInitEasierTaskCloseInitEnv',
        max_episode_steps=100,
        )

class FetchHookSweepPushDenseEasyInitEasierTaskCloseInitNoGraspEnv(FetchHookSweepPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=False, easier_task=True, close_push_init=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookSweepPushDenseEasyInitEasierTaskCloseInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepPushDenseEasyInitEasierTaskCloseInitNoGraspEnv',
        max_episode_steps=100,
        )

class FetchHookSweepPushDenseEasyInitEasierTaskCloseInitNoGraspEnvRandomized(FetchHookSweepPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=False, easier_task=True, close_push_init=True, grasp_reward=False, easy_init=True,
                        randomize_cube=True)

register(
        id='FetchHookSweepPushDenseEasyInitEasierTaskCloseInitNoGraspRandomized-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepPushDenseEasyInitEasierTaskCloseInitNoGraspEnvRandomized',
        max_episode_steps=100,
        )



class FetchHookSweepPushBetterDenseEasyInitEasierTaskCloseInitNoGraspEnv(FetchHookSweepPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easier_task=True, close_push_init=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookSweepPushBetterDenseEasyInitEasierTaskCloseInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepPushBetterDenseEasyInitEasierTaskCloseInitNoGraspEnv',
        max_episode_steps=100,
        )

class FetchHookSweepPushBetterDenseEasyInitEasierTaskCloseInitEnv(FetchHookSweepPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, easier_task=True, close_push_init=True, grasp_reward=True, easy_init=True)

register(
        id='FetchHookSweepPushBetterDenseEasyInitEasierTaskCloseInit-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepPushBetterDenseEasyInitEasierTaskCloseInitEnv',
        max_episode_steps=100,
        )

class FetchHookSweepPushBetterDenseEasyInitSideInitNoGraspEnv(FetchHookSweepPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, cube_side_init=True, easier_task=True, close_push_init=True, grasp_reward=False, easy_init=True)

register(
        id='FetchHookSweepPushBetterDenseEasyInitSideInitNoGrasp-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepPushBetterDenseEasyInitSideInitNoGraspEnv',
        max_episode_steps=100,
        )


class FetchHookSweepPushBetterDenseEasyInitSideInitEnv(FetchHookSweepPushEnv):
    def __init__(self):
        super().__init__(sparse=False, better_dense_reward=True, cube_side_init=True, easier_task=True, close_push_init=True, grasp_reward=True, easy_init=True)

register(
        id='FetchHookSweepPushBetterDenseEasyInitSideInit-v0',
        entry_point='rl_with_teachers.envs:FetchHookSweepPushBetterDenseEasyInitSideInitEnv',
        max_episode_steps=100,
        )
