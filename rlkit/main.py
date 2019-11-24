import argparse

import random
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv
from gym import spaces
from gym.envs.robotics.fetch import pick_and_place

import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import torch_ify, np_ify, eval_np
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.data_management.demo_buffer import DemoBuffer

import robosuite

class OptimalPickPlaceAgent(TeacherPolicy):
    """
    For Fetch reach environment with single start and goal. This agent
    reaches for the goal position unconditionally.
    """
    def __init__(self, goal, noise=None):
        self.goal = goal
        self.noise = noise
        self.picker = PickAgent(return_to_start=True)
        self.placer = PlaceAgent(goal)

    def __call__(self, obs):
        cube_pos = obs[3:6]
        goal_pos = obs[-6:-3]
        achieved = np.linalg.norm(goal_pos - cube_pos) < 0.01
        if not self.picker.is_grasping(obs) and not achieved:
            action = self.picker(obs)
        else:
            action = self.placer(obs)

        # noisy_action = self.apply_noise(np.array(action))
        noisy_action = np.array(action)
        noisy_action[-1] = action[-1]#0.95*action[-1]+0.05*noisy_action[-1]
        return noisy_action

    def reset(self):
        self.picker.reset()
        self.placer.reset()

class PickAgent(TeacherPolicy):
    """
    For Fetch PickAndPlace environment with single cube position and goal.
    This agent always tries to pick up the cube in a fashion similar
    to a claw machine.

    It first moves translationally into place before moving vertically towards
    the target. Once it is sufficiently close, the agent attempts
    to grab the target object.
    """
    def __init__(self, noise=None, return_to_start=False):
        self.target_close_to_grasp = 0.01
        self.target_close_enough = 0.005
        self.lateral_close_enough = 0.005
        self.noise = noise
        self.object_gripper_pos = None

        # whether to return to approximate arm starting position after grasping
        self.return_to_start = return_to_start

    def is_close_to_grasping(self, obs, obj_index=3):
        gripper_pos = obs[:3]
        object_pos = obs[obj_index:obj_index + 3]
        delta_x = object_pos - gripper_pos
        distance_to_target = np.linalg.norm(delta_x)
        return distance_to_target < self.target_close_to_grasp

    def is_grasping(self, obs, obj_index=3):
        gripper_pos = obs[:3]
        object_pos = obs[obj_index:obj_index + 3]
        delta_x = object_pos - gripper_pos
        distance_to_target = np.linalg.norm(delta_x)
        return distance_to_target < self.target_close_enough and obs[10] < 0.03

    def __call__(self, obs, obj_index=3):
        gripper_pos = obs[:3]
        object_pos = obs[obj_index:obj_index + 3]
        delta_x = object_pos - gripper_pos
        distance_to_target = np.linalg.norm(delta_x)
        max_lateral_coordinate_deviation = max(np.abs(delta_x[:2]))
        robot_is_grasping = self.is_grasping(obs, obj_index)
        if self.is_close_to_grasping(obs, obj_index) and not robot_is_grasping:
            # attempt a grasp if close enough.
            gripper_action = -1.0
            delta_x /= 0.05
        else:
            self.object_gripper_pos = None
            gripper_action = 1.

            if self.return_to_start and robot_is_grasping:
                # if already grasping, try returning to cube start position
                gripper_action = -1.0
                from rl_with_teachers.envs import FetchOneGoalPickPlaceEnv
                cube_start_position = np.array(FetchOneGoalPickPlaceEnv.OBJECT_START_WITH_Z)
                delta_x = cube_start_position - gripper_pos
                distance_to_target = np.linalg.norm(delta_x)
                max_lateral_coordinate_deviation = max(np.abs(delta_x[:2]))
                z_coordinate_deviation = np.abs(delta_x[2])

                if max_lateral_coordinate_deviation < 0.05:
                    # apply action as actual lateral distance to target
                    delta_x[:2] /= 0.05
                else:
                    delta_x[:2] /= (max_lateral_coordinate_deviation*2.0)

                if z_coordinate_deviation < 0.05:
                    # apply action as actual vertical distance to target
                    delta_x[2] /= 0.05
                else:
                    delta_x[2] /= z_coordinate_deviation
            else:
                if max_lateral_coordinate_deviation < self.lateral_close_enough:
                    # move vertically
                    delta_x[:2] = 0.
                    z_coordinate_deviation = np.abs(delta_x[2])
                    if z_coordinate_deviation < 0.05:
                        # apply action as actual vertical distance to target
                        delta_x[2] /= 0.05
                    else:
                        delta_x[2] /= z_coordinate_deviation
                else:
                    # move laterally
                    delta_x[2] = 0.
                    if max_lateral_coordinate_deviation < 0.05:
                        # apply action as actual lateral distance to target
                        delta_x[:2] /= 0.05
                    else:
                        # compute a direction heading towards the can, but normalize so that
                        # the largest component of the vector is 1 or -1
                        # (the largest allowed delta action, which is a distance of 0.05)
                        delta_x[:2] /= (max_lateral_coordinate_deviation*2.0)

        action = np.concatenate([delta_x, [gripper_action]])
        # noisy_action = self.apply_noise(np.array(action))
        noisy_action = np.array(action)
        noisy_action[-1] = action[-1]#0.95*action[-1]+0.05*noisy_action[-1]
        return noisy_action


class PlaceAgent(TeacherPolicy):
    """
    For Fetch PickAndPlace environment with single cube position and goal.
    This agent assumes that it already has the cube grasped and just follows a
    simple reaching policy to move the cube to the goal.
    """
    def __init__(self, goal, noise=None, axis_aligned=False, use_gripper=True, place_parabolic=True):
        """
        @axis_aligned: align x before aligning y
        @use_gripper: if False, don't release the object after moving it into place
        @place_parabolic: if True, place the object using a parabolic arc,
            otherwise just place it normally, with a bit of positive z-offset
        """
        self.goal = goal
        self.target_close_enough = 0.01
        self.object_close_enough = 0.075
        self.noise = noise

        # NOTE: if this is true, x is aligned before y
        self.axis_aligned = axis_aligned

        self.use_gripper = use_gripper
        self.place_parabolic = place_parabolic

    def placed_close_enough(self, obs):
        gripper_pos = obs[:3]
        delta_x = np.array(self.goal) - gripper_pos

        return np.abs(delta_x[0]) < self.target_close_enough and np.abs(delta_x[1]) < self.target_close_enough

    def __call__(self, obs, obj_index=3):
        gripper_pos = obs[:3]
        object_pos = obs[obj_index:obj_index + 3]
        delta_to_obj = object_pos - gripper_pos
        distance_to_object = np.linalg.norm(delta_to_obj)

        delta_x = np.array(self.goal) - gripper_pos
        if self.axis_aligned and not (np.abs(self.goal[0] - gripper_pos[0]) < self.target_close_enough):
            delta_x[1] = 0

        max_coordinate_deviation = np.max(delta_x)

        if self.placed_close_enough(obs):
            if self.use_gripper:
                gripper_act = 1.
            else:
                gripper_act = -1.
            action = np.array([0., 0., 0., gripper_act])
        else:
            action = np.array([0., 0., 0., -1.])

            ### TODO (Andrey): Why is this normalization weird? shouldn't it just be the max deviation to get actions of 1?
            delta_x /= (max_coordinate_deviation*2+0.1)

            ### TODO (Andrey): is it okay if I make this change? unclear what the z-action is doing... ###

            if self.place_parabolic:
                # take parabolic arc to place the object
                action[:3] = delta_x + np.array([0.0,0.0,max_coordinate_deviation-delta_x[2]])
            else:
                # just hover a little bit
                action[:3] = delta_x + np.array([0.0,0.0,0.05])

            # if self.use_parabolic:
            #     pass
            # action[:3] = delta_x + np.array([0.0,0.0,max_coordinate_deviation-delta_x[2]])
            # action[:3] = delta_x + np.array([0.0,0.0,0.05])

        # noisy_action = self.apply_noise(np.array(action))
        noisy_action = np.array(action)
        noisy_action[-1] = action[-1]#0.95*action[-1]+0.05*noisy_action[-1]
        return noisy_action

class FetchOneGoalPickPlaceEnv(pick_and_place.FetchPickAndPlaceEnv):
    """
    A mujoco task with a fetch robot that needs to pick up a cube and move it to a
    goal point (or just push it there).
    """
    OBJECT_START = np.array([1.25, 0.55])
    OBJECT_START_WITH_Z = np.array([1.25, 0.55, 0.425])
    ROBOT_START = np.array([1.34184371, 0.75, 0.5])
    def __init__(self,
                 sparse=True,
                 goal=np.array([1.45, 0.55, 0.425]),
                 better_dense_reward=False):
        self.goal = goal
        self.better_dense_reward = better_dense_reward
        super().__init__('sparse' if sparse else 'dense')
        self.is_sparse = sparse

        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype='float32')

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.is_sparse:
            if reward == 0:
                reward = 1.0
            else:
                reward = 0.0
            reward/=10.0
        elif self.better_dense_reward:
            # Better dense reward is different from default Fetch
            # dense reward of l2; ours increades from 0 to 1
            # as object nears goal
            obj_pos = obs['observation'][3:6]
            obj_goal = obs['desired_goal']

            ### give dense reward for distance from block to goal
            goal_dist = np.linalg.norm(obj_goal - obj_pos)
            goal_reward = 1. - np.tanh(10.0 * goal_dist)
            reward = goal_reward

        object_goal = np.array(obs['desired_goal'])
        obs_to_ret = np.concatenate([obs['observation'], object_goal])
        return obs_to_ret, reward, done, info

    def reset(self):
        obs = super().reset()
        object_goal = np.array(obs['desired_goal'])
        obs_to_ret = np.concatenate([obs['observation'], object_goal])
        return obs_to_ret

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        object_xpos = np.array(FetchOneGoalPickPlaceEnv.OBJECT_START)
        object_xpos[0]+=(random.random()-0.5)/10.0
        object_xpos[1]+=(random.random()-0.5)/10.0
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()

        gripper_offset = (np.random.random(3)-0.5)/10
        gripper_offset[2] = gripper_offset[2]/2
        gripper_target = self.sim.data.get_site_xpos('robot0:grip') + gripper_offset
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()
        return True

    def _sample_goal(self):
        goal = self.goal.copy()
        return goal

    def make_teachers(self):
        return [OptimalPickPlaceAgent(self.goal, noise)]

# register(
#         id='OneGoalPickPlaceEnv-v0',
#         entry_point='rl_with_teachers.envs:FetchOneGoalPickPlaceEnv',
#         max_episode_steps=100,
#         )

class ACTeachBehaviorPolicy:
    """
    Simple behavioral policy to pick actions based on Q-values from learner.
    """
    def __init__(self, learner, teachers, value_function):
        self.learner = learner
        self.teachers = teachers
        self.value_function = value_function
        self.current_policy_choice = None

    def get_action(self, obs):
        """
        Takes a single numpy observation and returns a single action
        """

        # gather action proposals from learner and teachers for current observation
        learner_action, _ = self.learner.get_action(obs)
        actions = [learner_action]
        for teacher in self.teachers:
            action = teacher.get_action(obs)
            actions.append(action)
        actions = np.array(actions)

        # score each action using value function and pick the best one
        obs_stacked = np.tile(obs, (len(actions), 1))
        values = eval_np(module=self.value_function, obs_stacked, actions)
        max_q_choice = np.argmax(values)
        self.current_policy_choice = max_q_choice
        return actions[max_q_choice], {}

    def reset(self):
        self.current_policy_choice = None

def create_robosuite_env(env_name, horizon=1000, render=False):

    # reward shaping configuration
    reward_shaping = True 

    env = robosuite.make(
        env_name,
        has_renderer=render,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=False,
        camera_height=84,
        camera_width=84,
        camera_name="agentview",
        gripper_visualization=False,
        reward_shaping=reward_shaping,
        control_freq=100,
        horizon=horizon,
    )

    return RobosuiteEnv(env)

class RobosuiteEnv:
    env = None

    def __init__(self, env, keys=None):
        """
        Initializes the Gym wrapper.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        self.env = env

        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "object-state"]
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        # reward range
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None

        self.robosuite = True

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()

        # reset arm configuration to line up with demonstrations 
        self.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        ob_dict = self.env._get_observation()

        if self.env.has_renderer:
            self.env.viewer.set_camera(0)

        return self._flatten_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _check_success(self):
        return self.env._check_success()

    def close(self):
        self.env.close()

    def make_teachers(self, type, env, agent, noise=None):
        if type == 'optimal':
            return [IRISAgent(env=env, agent=agent, noise=noise)]
        else:
            raise ValueError('Not a valid teacher type '+type)

def experiment(variant):
    env_name = variant["env_name"]
    horizon = variant["horizon"]
    eval_horizon = variant["eval_horizon"]
    render = variant["render"]
    render_eval = variant["render_eval"]

    expl_env = create_robosuite_env(
        env_name=env_name, 
        horizon=horizon, 
        render=render,
    )
    eval_env = create_robosuite_env(
        env_name=env_name, 
        horizon=eval_horizon,
        render=render_eval 
    )
    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())

    # grab teachers from env
    teachers = expl_env.make_teachers()

    # create networks
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)

    behavioral_policy = ACTeachBehaviorPolicy(
        learner=policy,
        teachers=teachers,
        value_function=qf1,
    )

    eval_path_collector = MdpPathCollector(
        env=eval_env,
        policy=eval_policy,
        render=render_eval,
    )
    expl_path_collector = MdpPathCollector(
        env=expl_env,
        policy=behavioral_policy,
        render=render,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of experiment for logging and stuff
    parser.add_argument(
        "--name",
        type=str,
    )

    # name of environment
    parser.add_argument(
        "--env",
        type=str,
    )

    # horizon for exploration rollouts
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
    )

    # horizon for evaluation rollouts
    parser.add_argument(
        "--eval_horizon",
        type=int,
        default=1000,
    )

    # number of evaluation rollouts per epoch
    parser.add_argument(
        "--eval_rollouts",
        type=int,
        default=1,
    )

    # whether to render to screen during training (exploration)
    parser.add_argument(
        "--render",
        action='store_true',
    )

    # whether to render to screen during training (evaluation)
    parser.add_argument(
        "--render_eval",
        action='store_true',
    )
    args = parser.parse_args()

    # cuda support
    ptu.set_gpu_mode(torch.cuda.is_available())

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=(args.eval_rollouts * args.eval_horizon), #5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=args.horizon, #1000,
            max_eval_path_length=args.eval_horizon, #1000
            batch_size=256,
            experiment_name=args.name,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        env_name=args.env,
        horizon=args.horizon,
        eval_horizon=args.eval_horizon,
        render=args.render,
        render_eval=args.render_eval,
    )
    setup_logger(args.name, variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
