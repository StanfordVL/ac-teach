import os
import gym
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.utils import seeding
from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

try:
    import robosuite
    from robosuite.wrappers import Wrapper
    import batchRL
except:
    print("WARNING: could not import robosuite!")

from rl_with_teachers.teachers.base import IRISAgent

def make_envs(env_id, do_eval, seed, conf, normalize_observations=False, normalize_returns=False):
    # Create envs.
    env_params = conf.pop('env_params',{})

    # hacky support for Sawyer envs
    if env_id.startswith("Sawyer"):
        env = base_env = create_robosuite_env(env_id, horizon=1000)
    else:
        env = base_env = gym.make(env_id)

        if hasattr(base_env, 'env'):
            base_env = base_env.env
    for attr in env_params:
        setattr(base_env, attr, env_params[attr])
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)

    if normalize_observations or normalize_returns:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=normalize_observations, norm_reward=normalize_returns)

    if do_eval:
        # hacky support for Sawyer envs
        if env_id.startswith("Sawyer"):
            eval_env = base_eval_env = create_robosuite_env(env_id, horizon=1000)
        else:
            eval_env = base_eval_env = gym.make(env_id)

            if hasattr(base_eval_env, 'env'):
                base_eval_env = base_eval_env.env
        for attr in env_params:
            setattr(base_eval_env, attr, env_params[attr])
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'), allow_early_resets=True)
        eval_env.seed(seed)
        eval_env.base_env = base_eval_env
    else:
        base_eval_env = None
        eval_env = None
    env.base_env = base_env

    return base_env, env, base_eval_env, eval_env

def create_robosuite_env(env_name, horizon=1000):

    # use visual versions of environments for training by default
    env_name = env_name[:6] + "Visual" + env_name[6:]

    # reward shaping configuration
    reward_shaping = True # False

    env = robosuite.make(
        env_name,
        has_renderer=False,
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

        return self._flatten_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def make_teachers(self, type, env, agent, noise=None):
        if type == 'optimal':
            return [IRISAgent(env=env, agent=agent, noise=noise)]
        else:
            raise ValueError('Not a valid teacher type '+type)


