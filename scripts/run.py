import argparse
import logging
import os
import datetime
import time
import copy
import pickle

from mpi4py import MPI

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.ddpg.memory import Memory
from stable_baselines.ddpg import DDPG
from stable_baselines.deepq import DQN, LnMlpPolicy, MlpPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.common.misc_util import boolean_flag

from rl_with_teachers.envs.util import make_envs
from rl_with_teachers.behavior_policies import make_behavior_policy
from rl_with_teachers.learners import *
from rl_with_teachers.utils import init_logging, ExponentialSchedule, parse_noise_types, parse_conf

'''
This is the main script to run things with.

It can be used for training policies as well as evaluating already trained policies.

It accepts a lot of command line arguments, but the most important one is the path to the config file;
any other arguments can be provided in the config, and specifying them in command line just overrides the
value in the config file (if there is one).
'''


def run(ddpg_type, log_dir, env_id, normalize_observations, normalize_returns, seed,
          noise_type, do_eval, num_timesteps, log_interval, teach_behavior_policy, load_path, memory_limit, **conf):
    """
    run the training of DDPG

    :param env_id: (str) the environment ID
    :param seed: (int) the initial random seed
    :param noise_type: (str) the wanted noises ('adaptive-param', 'normal' or 'ou'), can use multiple noise type by
        seperating them with commas
    :param layer_norm: (bool) use layer normalization
    :param evaluation: (bool) enable evaluation of DDPG training
    :param conf (dict) extra keywords for the training.train function
    """
    # Configure things.
    save_exps = conf.pop('save_exps', None)
    base_env, env, base_eval_env, eval_env = make_envs(env_id, do_eval, seed, conf,
                              normalize_observations, normalize_returns)

    nb_actions = env.action_space.shape[-1]
    action_noise,param_noise = parse_noise_types(noise_type, nb_actions)

    teachers_conf = conf.pop('teachers', None)
    behavior_policy_params = conf.pop('behavior_policy_params', None)
    if ddpg_type=='Dropout DDPG':
        learner = DropoutBayesianDDPG(env,
                     eval_env=eval_env,
                     param_noise=param_noise,
                     action_noise=action_noise,
                     memory_limit=memory_limit,
                     tensorboard_log=log_dir,
                     num_timesteps_final=num_timesteps,
                     **conf)
    else:
        learner = DDPGwTeachers(FeedForwardPolicy,
                     env,
                     eval_env=eval_env,
                     param_noise=param_noise,
                     action_noise=action_noise,
                     memory_limit=memory_limit,
                     tensorboard_log=log_dir,
                     **conf)
    base_eval_env.agent = learner

    if load_path is not None:
        learner.load(load_path)

    behavior_policy = make_behavior_policy(env_id, base_env, learner, teach_behavior_policy, teachers_conf, behavior_policy_params)
    for eval_reward in learner.learn(num_timesteps, behavior_policy, log_interval=log_interval):
        yield eval_reward

    if save_exps:
        save_experiences(log_dir + '/experiences.pkl', learner.memory)

    env.close()
    if eval_env is not None:
        eval_env.close()

    learner.save(os.path.join(log_dir,'model'))
    tf.reset_default_graph()

def parse_args():
    """
    parse the arguments for DDPG training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str, default='cfg/train_no_teachers.yaml')
    parser.add_argument('--log-base-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--learner-type', type=str, choices=['DDPG','Q'], default=None)

    # Can optionally overwrite some params from config
    parser.add_argument('--env-id', type=str, default=None)
    parser.add_argument('--teacher_behavior_policy', type=str, default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)  # per MPI worker
    parser.add_argument('--reward-scale', type=float, default=None)
    parser.add_argument('--noise-type', type=str, default=None)
    parser.add_argument('--feature-extraction', type=str, default=None)
    parser.add_argument('--nb-train-steps', type=int, default=None)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=None)  # per epoch cycle and MPI worker
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--nb-eval-steps', type=int, default=None)  # per epoch cycle and MPI worker
    parser.add_argument('--log-interval', type=int, default=None)  # per epoch cycle and MPI worker

    #parser.add_argument('--demo-path', type=str, default='')

    parser.add_argument(
    "--load-from", type=str, help="load the saved model and optimizer at the beginning"
    )

    boolean_flag(parser, 'do_eval', default=None)
    boolean_flag(parser, 'render-eval', default=None)
    boolean_flag(parser, 'render', default=None)
    boolean_flag(parser, 'save_exps', default=None)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    config_paths = args.pop('config')
    conf = parse_conf(config_paths, args)

    log_dir = args.pop('log_dir')
    if 'log_base_dir' in conf:
        log_base_dir = conf.pop('log_base_dir')
    else:
        log_base_dir = 'logs'

    experiment_name = conf.pop('experiment_name',None)
    take_next_seed = conf['seed'] == 'next'
    if take_next_seed:
        conf['seed'] = 0
    rank = MPI.COMM_WORLD.Get_rank()
    conf['seed']+=rank

    if log_dir is None:
        conf_name = config_paths[0].split('/')[-1][:-5]
        if not experiment_name:
            experiment_name = conf_name

        log_dir = '%s/%s/%s/seed_%d'%(log_base_dir,conf['env_id'],experiment_name,conf['seed'])
        while os.path.exists(log_dir):
            if take_next_seed:
                conf['seed']+=1
                log_dir = '%s/%s/%s/seed_%d'%(log_base_dir,conf['env_id'],experiment_name,conf['seed'])
            else:
                log_dir = '%s_%s'%(log_dir,str(datetime.datetime.now())[5:-10]\
                        .replace(' ','_').replace(':','_'))

    init_logging(log_dir)
    logger.configure(log_dir, ['stdout', 'log', 'csv', 'tensorboard'])
    logging.info('Running with config:\n%s'%str(conf))

    learner_type = conf.pop('learner_type')

    start_time = time.time()
    if learner_type == 'DDPG' or learner_type=='Dropout DDPG':
        for eval_reward in run(learner_type, logger.get_dir(), **conf):
            pass
    else:
        raise ValueError('%s not a valid learner type'%learner_type)
    logger.info('total runtime: {}s'.format(time.time() - start_time))
