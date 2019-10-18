import os
import gym
import tensorflow as tf
from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

def make_envs(env_id, do_eval, seed, conf, normalize_observations=False, normalize_returns=False):
    # Create envs.
    env_params = conf.pop('env_params',{})
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
