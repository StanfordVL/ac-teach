from abc import ABC, abstractmethod
from stable_baselines.ddpg.ddpg import *
from rl_with_teachers.utils import log_histogram, log_images
import logging
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
from stable_baselines import logger, deepq
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger
from stable_baselines.common import Dataset, explained_variance, fmt_row, zipsame
from stable_baselines.common.schedules import LinearSchedule
import stable_baselines.common.tf_util as U
from collections import defaultdict
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.tensor_util import constant_value
from gym.spaces import Box
from scipy.stats import invgamma
from gym.spaces import Discrete

class RLwTeachersLearner(ABC):
    """ An abstract class to represent an RL policy to be taught by policy teachers
    """
    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, obs, with_exploration=False):
        """
        The policy act method.

        Should be able to return the action, and also return a Qvalue (as a 2-tuple).
        If compute_q is False, return None for q_value.

        :param obs: The environment observation, as a numpy array
        :param with_exploration: Whether to generate actions with exploration
        """
        pass

    def step_learn(self, obs, new_obs, reward, done):
        """ Optional learn method """
        pass

class DDPGwTeachers(DDPG, RLwTeachersLearner):
    """
    Deep Deterministic Policy Gradient (DDPG) model.
    Largely same as Stable Baselines (https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ddpg/ddpg.py)
    implementation, main changes are noted with CHANGES.

    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    :param policy: (DDPGPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param teachers: (list) list of functions ; should be callable with one param, obs, to get action
    :param gamma: (float) the discount factor
    :param memory_policy: (Memory) the replay buffer (if None, default to baselines.ddpg.memory.Memory)
    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evalutation steps
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param normalize_returns: (bool) should the critic output be normalized
    :param enable_popart: (bool) enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf)
    :param normalize_observations: (bool) should the observation be normalized
    :param batch_size: (int) the size of the batch for learning the policy
    :param observation_range: (tuple) the bounding values for the observation
    :param return_range: (tuple) the bounding values for the critic output
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param clip_norm: (float) clip the gradients (disabled if None)
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param render_eval: (bool) enable rendering of the evalution environment
    :param memory_limit: (int) the max number of transitions to store
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """
    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., action_l2=0.0,
                 render=False, render_eval=False, memory_limit=50000, verbose=0, tensorboard_log=None, use_meta_target=False,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):
        self.use_meta_target = use_meta_target
        self.action_l2 = action_l2

        # variables for model saving
        self.last_model_save_time = -1.
        self.hours_elapsed = 0

        super(DDPGwTeachers, self).__init__(policy, env, gamma, memory_policy, eval_env, nb_train_steps,
                 nb_rollout_steps, nb_eval_steps, param_noise, action_noise,
                 normalize_observations, tau, batch_size, param_noise_adaption_interval,
                 normalize_returns, enable_popart, observation_range, critic_l2_reg,
                 return_range, actor_lr, critic_lr, clip_norm, reward_scale,
                 render, render_eval, memory_limit, verbose, tensorboard_log,
                 _init_setup_model, policy_kwargs, full_tensorboard_log)


    def setup_model(self):
         super().setup_model() # See below in DropoutBayesianDDPG for full model set up

         # CHANGES - if running with meta target, also need to set up placeholders
         if self.use_meta_target:
             self.setup_meta_target()

    def setup_meta_target(self):
        """
        New function to set up meta target TF ops
        """
        with self.graph.as_default():
            clipped_obs1 = tf.clip_by_value(self.target_policy.processed_obs,
                                           self.observation_range[0], self.observation_range[1])

            with tf.variable_scope("meta_target", reuse=False):
                critic_target = self.target_policy.make_critic(clipped_obs1, self.actions1)

            with tf.variable_scope("loss", reuse=False):
                q_obs1 = critic_target
                self.target_q_no_actor = self.rewards + (1. - self.terminals1) * self.gamma * q_obs1

            with self.sess.as_default():
                self._initialize(self.sess)

    def _train_step(self, step, writer, log=False):
        """
        run a step of training from batch
        :param step: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param log: (bool) whether or not to log to metadata
        :return: (float, float) critic loss, actor loss
        """
        if not self.use_meta_target:
            # Super class implements normal training step
            return super()._train_step(step, writer, log)

        batch = self.memory.sample(batch_size=self.batch_size)
        # CHANGES - choose action based on chosen policy
        actions1 = self.learn_behavior_policy.choose_actions_greedy(batch['obs1'])[1]

        target_q = self.sess.run(self.target_q, feed_dict={
            self.action_target: actions1,
            self.obs_target: batch['obs1'],
            self.rewards: batch['rewards'],
            self.terminals1: batch['terminals1'].astype('float32')
        })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        td_map = {
            self.obs_train: batch['obs0'],
            self.actions: batch['actions'],
            self.action_train_ph: batch['actions'],
            self.rewards: batch['rewards'],
            self.critic_target: target_q,
            self.param_noise_stddev: 0 if self.param_noise is None else self.param_noise.current_stddev
        }
        if writer is not None:
            # run loss backprop with summary if the step_id was not already logged (can happen with the right
            # parameters as the step value is only an estimate)
            if self.full_tensorboard_log and log and step not in self.tb_seen_steps:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, actor_grads, actor_loss, critic_grads, critic_loss = \
                    self.sess.run([self.summary] + ops, td_map, options=run_options, run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%d' % step)
                self.tb_seen_steps.append(step)
            else:
                summary, actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run([self.summary] + ops,
                                                                                            td_map)
            writer.add_summary(summary, step)
        else:
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, td_map)

        self.actor_optimizer.update(actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer.update(critic_grads, learning_rate=self.critic_lr)

        return critic_loss, actor_loss

    def eval_q_value(self, obs, action=None):
        """
        Evaluates a q value from the critic. If action is None, evaluates actor's action.
        """
        if action is None:
            return self.sess.run(self.critic_with_actor_tf,  {self.obs_train: obs})
        else:
            return self.sess.run(self.critic_tf,  {self.obs_train: obs, self.actions: action})

    def eval_actor(self, obs):
        """
        Computes the actions of the actor for given observation.
        """
        feed_dict = {self.obs_train: obs}
        action = self.sess.run(self.actor_tf, feed_dict=feed_dict)
        return action

    def _policy(self, obs, apply_noise=True, compute_q=True):
        """
        Get the agent actor and critic output, from a given observation.
        no change from stable baselines version, here for easy readability.
        :param obs: ([float] or [int]) the observation
        :param apply_noise: (bool) enable the noise
        :param compute_q: (bool) compute the critic output
        :return: ([float], float) the action and critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)
        feed_dict = {self.obs_train: obs}
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
            feed_dict[self.obs_noise] = obs
        else:
            actor_tf = self.actor_tf

        if compute_q:
            action, q_value = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q_value = None

        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, -1, 1)
        return action, q_value

    def select_action(self, obs, with_exploration=False):
        # Helper function that just calls _policy and returns action without q
        return self._policy(obs, apply_noise=with_exploration, compute_q=True)[0]

    def _setup_target_network_updates(self):
        """
        set the target update operations.
        no change from stable baselines version, here for easy readability.
        """
        init_updates, soft_updates = get_target_updates(tf_util.get_trainable_vars('model/'),
                                                        tf_util.get_trainable_vars('target/'), self.tau,
                                                        self.verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

    def _setup_critic_optimizer(self):
        """
        setup the optimizer for the critic
        no change from stable baselines version, here for easy readability.
        """
        if self.verbose >= 2:
            logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in tf_util.get_trainable_vars('model/qf/')
                               if 'bias' not in var.name and 'output' not in var.name and 'b' not in var.name]
            if self.verbose >= 2:
                for var in critic_reg_vars:
                    logger.info('  regularizing: {}'.format(var.name))
                logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = tf_util.flatgrad(self.critic_loss, tf_util.get_trainable_vars('model/qf/'),
                                             clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/qf/'), beta1=0.9, beta2=0.999,
epsilon=1e-08)

    def _setup_actor_optimizer(self):
        """
        Setup the optimizer for the actor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        if self.action_l2:
            # CHANGES - add option to have l2 loss on actions
            self.actor_loss+= self.action_l2 * tf.reduce_mean(tf.square(self.actor_tf))

        actor_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf_util.flatgrad(self.actor_loss, tf_util.get_trainable_vars('model/pi/'),
                                            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/pi/'), beta1=0.9, beta2=0.999,
            epsilon=1e-08)

    def learn(self, total_timesteps, behavior_policy=None, callback=None, seed=None, log_interval=100, tb_log_name="DDPG",
              reset_num_timesteps=True, tb_write_extra=False):
        """
        The learning loop.
        Largely same as normal DDPG, but with addition of actions suggestions from teachers.
        """
        self.learn_behavior_policy = behavior_policy
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        if behavior_policy is None:
            num_teachers = 0
        else:
            num_teachers = len(behavior_policy.teachers)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log)\
                as writer:
            if not tb_write_extra:
                writer = None

            self._setup_learn(seed)

            # a list for tensorboard logging, to prevent logging with the same step number, if it already occured
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()
            # we assume symmetric actions.
            assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)
            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            self.episode_reward = np.zeros((1,))
            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                obs = self.env.reset()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs = self.eval_env.reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_agent_choices = []
                epoch_agent_actions = [[] for i in range(1+num_teachers)]
                epoch_agent_q_vals = [[] for i in range(1+num_teachers)]
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0

                logging.info('Starting DDPG training')
                while True:
                    logging.info('----------------------------------------------')
                    logging.info('Starting new episode')
                    first_rollout = True
                    first_rollout_eval = True
                    for _ in range(log_interval):
                        # Perform rollouts.
                        discount = 1.0
                        for _ in range(self.nb_rollout_steps):
                            if total_steps >= total_timesteps:
                                return self

                            # Predict next action.
                            if behavior_policy is not None:
                                # CHANGES - this is where the behavioral policy is used to select action
                                policy_choice, action, q_value = behavior_policy.choose_action(obs)
                            else:
                                policy_choice = 0
                                action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                            if q_value is None:
                                q_value = 0.0
                            assert action.shape == self.env.action_space.shape

                            # Execute next action.
                            new_obs, reward, done, _ = self.env.step(action * np.abs(self.action_space.low))
                            logging.info('At state %s chose action %s from policy %d for reward of %f'%(str(obs), str(action), policy_choice, reward))

                            if writer is not None:
                                ep_rew = np.array([reward]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                                                  writer, self.num_timesteps)
                            step += 1
                            total_steps += 1
                            self.num_timesteps += 1
                            if self.render and first_rollout:
                                self.env.render()
                            episode_reward += discount*reward
                            discount*=self.gamma
                            episode_step += 1

                            # Book-keeping.
                            epoch_agent_choices.append(policy_choice)
                            epoch_agent_actions[policy_choice].append(action)
                            epoch_agent_q_vals[policy_choice].append(q_value)
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)
                            self._store_transition(obs, action, reward, new_obs, done)
                            if behavior_policy is not None:
                                behavior_policy.step_learn(obs, policy_choice, reward, new_obs, done)
                            obs = new_obs
                            if callback is not None:
                                # Only stop training if return value is False, not when it is None.
                                # This is for backwards compatibility with callbacks that have no return statement.
                                if callback(locals(), globals()) is False:
                                    return self

                            if done:
                                logging.info('Episode finished! num steps=%d, final reward=%f'%(episode_step, episode_reward))
                                #first_rollout = False

                                # Episode done.
                                epoch_episode_rewards.append(episode_reward)
                                episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)
                                episode_reward = 0.
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1
                                discount = 1.0

                                self._reset()
                                if not isinstance(self.env, VecEnv):
                                    obs = self.env.reset()

                                if behavior_policy is not None:
                                    behavior_policy.reset()


                        logging.info('----------------------------------------------')
                        logging.info('Starting agent training!')
                        # Train.
                        epoch_actor_losses = []
                        epoch_critic_losses = []
                        epoch_adaptive_distances = []
                        for t_train in range(self.nb_train_steps):
                            if self.memory.nb_entries >= self.batch_size and \
                                    t_train % self.param_noise_adaption_interval == 0:
                                distance = self._adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            # weird equation to deal with the fact the nb_train_steps will be different
                            # to nb_rollout_steps
                            step = (int(t_train * (self.nb_rollout_steps / self.nb_train_steps)) +
                                    self.num_timesteps - self.nb_rollout_steps)

                            critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)
                            logging.info('Train step %d, critic mean loss=%f , actor mean loss=%f'%\
                                    (t_train,np.mean(critic_loss),np.mean(actor_loss)))
                            epoch_critic_losses.append(critic_loss)
                            epoch_actor_losses.append(actor_loss)
                            self._update_target_net()
                        logging.info('Finished agent training!')

                        # Evaluate.
                        eval_episode_rewards = []
                        eval_qs = []
                        if self.eval_env is not None:
                            eval_episode_reward = 0.0
                            eval_steps = 0
                            discount = 1.0
                            self.eval_env.reset()
                            for _ in range(self.nb_eval_steps):
                                if total_steps >= total_timesteps:
                                    return self
                                if self.render_eval and first_rollout_eval:
                                    self.eval_env.render()
                                eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                                eval_obs, eval_r, eval_done, _ = self.eval_env.step(eval_action *
                                                                                    np.abs(self.action_space.low))
                                eval_steps+=1
                                eval_episode_reward += discount*eval_r
                                discount*=self.gamma

                                eval_qs.append(eval_q)
                                if eval_done:
                                    logging.info('Eval episode finished. num steps=%d, final reward=%f'%(eval_steps, eval_episode_reward ))
                                    if self.render_eval and first_rollout_eval:
                                        self.eval_env.render()
                                    eval_obs = self.eval_env.reset()
                                    first_rollout_eval = False
                                    yield eval_episode_reward
                                    eval_episode_rewards.append(eval_episode_reward)
                                    eval_episode_rewards_history.append(eval_episode_reward)
                                    eval_episode_reward = 0.
                                    discount = 1.0
                                    eval_steps=1

                        # Save the model every hour.
                        if time.time() - self.last_model_save_time > 3600:
                            self.save(os.path.join(logger.get_dir(),'model_hour_{}'.format(self.hours_elapsed)))
                            self.last_model_save_time = time.time()
                            self.hours_elapsed += 1

                    combined_stats = {}
                    if self.nb_rollout_steps > 0:
                        duration = time.time() - start_time
                        stats = self._get_stats()
                        for key in sorted(stats.keys()):
                            if not ('Q_mc' in key):
                                combined_stats[key] = stats[key]
                            else:
                                log_histogram(key, stats[key], step, 25)
                        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                        if epoch_episode_rewards:
                            log_histogram('rollout/return', epoch_episode_rewards, step, 25)
                        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                        if episode_rewards_history:
                            log_histogram('rollout/return', episode_rewards_history, step, 25)
                        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                        combined_stats['rollout/agent_choice_mean'] = np.mean(epoch_agent_choices)
                        log_histogram('rollout/agent_choices', epoch_agent_choices, step, 25)
                        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                        for i in range(num_teachers + 1):
                            if len(epoch_agent_actions[i]) > 25:
                                log_histogram('rollout/actions/%d'%i, epoch_agent_actions[i], step, 25)
                                log_histogram('rollout/Q/%d'%i, epoch_agent_q_vals[i], step, 25)
                        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                        if len(epoch_adaptive_distances) != 0:
                            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                        combined_stats['total/duration'] = duration
                        combined_stats['total/steps_per_second'] = float(step) / float(duration)
                        combined_stats['total/episodes'] = episodes
                        combined_stats['rollout/episodes'] = epoch_episodes
                        combined_stats['rollout/actions_std'] = np.std(epoch_actions)

                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = np.mean(eval_qs)
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)


                    # Total statistics.
                    combined_stats['total/epochs'] = epoch + 1
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    logger.dump_tabular()
                    logger.info('')

        logging.info('Finished DDPG training')

class DropoutBayesianDDPGModels(FeedForwardPolicy):
    """
    Class that implements actor critic models for DDPG, using a MLP (2 layers of 64), with layer normalisation
    and dropout layers that allow for bayesian estimates of critic values

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", layer_norm=False, mc_samples=50,
                 dropout_keep_prob=0.99, merge_layer = 1, **kwargs):
        FeedForwardPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, layers,
                cnn_extractor, feature_extraction, layer_norm)
        self.mc_samples = mc_samples
        self.nb_actions = self.ac_space.shape[0]
        self.keep_prob = dropout_keep_prob
        self.merge_layer = merge_layer
        self.hidden_sizes = self.layers

    def make_critic(self, obs=None, action=None, reuse=False, scope="critic"):
        """
        Builds the critic, given an op or a placeholder for observation and action
        """
        self.num_qs = 0
        # Make the critic without dropout
        qf = self.make_qf(obs, action, reuse, scope, include_dropout=False)
        # Make N=mc_samples versions of critic with dropout layers, for bayesian estimation
        # Each instance has a dropout layer that masks output independently, so in effect we get N different outputs
        # These N monte carlo samples are then used to estimate mean and variance of the prediction
        # Having N ops with different outputs allows for computing N outputs at once, instead of in a loop
        qf_mc = [self.make_qf(obs, action, True, scope, include_dropout=True) for i in range(self.mc_samples)]

        return qf, qf_mc

    def make_qf(self, obs, action, reuse, scope, include_dropout=False):
        # Helper function to make multiple critics that share all weights but have independent dropout masks
        self.num_qs+=1
        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                qf_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                qf_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)
                if include_dropout:
                    # We do dropout after activation, at each layer
                    qf_h = tf.layers.dropout(qf_h, rate=1-self.keep_prob, seed=self.num_qs, name='drop_%d'%self.num_qs,
                                             training=True)
                if i == 0:
                    qf_h = tf.concat([qf_h, action], axis=-1)

                qf = tf.layers.dense(qf_h, 1,
                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                      maxval=3e-3))
        return qf

def logsumexp(x, axis=None):
    # Helper func used for bayesian dropout estimation
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def get_vars(scope, trainable=False):
    # Helper func for getting vars
    if trainable:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    else:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

class DropoutBayesianDDPG(DDPGwTeachers):
    """
    Adapted from https://github.com/Breakend/BayesianPolicyGradients
    Deep Deterministic Policy Gradient (DDPG) model
    DDPG: https://arxiv.org/pdf/1509.02971.pdf
    :param policy: (DDPGPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param memory_policy: (Memory) the replay buffer (if None, default to baselines.ddpg.memory.Memory)
    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evalutation steps
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param normalize_returns: (bool) should the critic output be normalized
    :param enable_popart: (bool) enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf)
    :param batch_size: (int) the size of the batch for learning the policy
    :param observation_range: (tuple) the bounding values for the observation
    :param return_range: (tuple) the bounding values for the critic output
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param clip_norm: (float) clip the gradients (disabled if None)
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param render_eval: (bool) enable rendering of the evalution environment
    :param memory_limit: (int) the max number of transitions to store
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    """

    def __init__(self, env, name='agent', gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., action_l2=0.0,
                 dropout_tau=0.85, num_timesteps_final=1e6, length_scale=0.01, is_teacher=False,
                 render=False, render_eval=False, memory_limit=100, verbose=0, tensorboard_log=None, use_meta_target=False,
                 include_mc_stats=True,_init_setup_model=True, policy_kwargs=None):

        self.dropout_tau = dropout_tau
        self.num_timesteps_final = num_timesteps_final
        self.length_scale = length_scale
        self.is_teacher = is_teacher
        self.name = name
        self.include_mc_stats = include_mc_stats
        super().__init__(DropoutBayesianDDPGModels, env, gamma, memory_policy, eval_env, nb_train_steps,
                 nb_rollout_steps, nb_eval_steps, param_noise, action_noise,
                 False, tau, batch_size, param_noise_adaption_interval,
                 False, enable_popart, observation_range, critic_l2_reg,
                 return_range, actor_lr, critic_lr, clip_norm, reward_scale, action_l2,
                 render, render_eval, memory_limit, verbose, tensorboard_log, use_meta_target,
                 _init_setup_model, policy_kwargs)

    def setup_model(self):
        logging.info('Starting DDPG model setup')
        # This is mostly same as in stable baselines, with a few changes marked with CHANGES
        with SetVerbosity(self.verbose):

            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: DDPG cannot output a {} action space, only spaces.Box is supported.".format(self.action_space)
            assert issubclass(self.policy, DDPGPolicy), "Error: the input policy for the DDPG model must be " \
                                                        "an instance of DDPGPolicy."
            self.graph = tf.get_default_graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                if not self.is_teacher:
                    self.memory = self.memory_policy(limit=self.memory_limit,
                                                     action_shape=self.action_space.shape,
                                                     observation_shape=self.observation_space.shape)

                with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                 **self.policy_kwargs)

                    # Create target networks.
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                     **self.policy_kwargs)
                    self.obs_target = self.target_policy.obs_ph
                    self.action_target = self.target_policy.action_ph

                    obs0 = tf.clip_by_value(self.policy_tf.processed_obs,
                                                       self.observation_range[0], self.observation_range[1])
                    obs1 = tf.clip_by_value(self.target_policy.processed_obs,
                                                       self.observation_range[0], self.observation_range[1])

                    if self.param_noise is not None:
                        # Configure perturbed actor for better exploration
                        # https://openai.com/blog/better-exploration-with-parameter-noise/
                        self.param_noise_actor = self.policy(self.sess, self.observation_space, self.action_space, 1, 1,
                                                             None, **self.policy_kwargs)
                        self.obs_noise = self.param_noise_actor.obs_ph
                        self.action_noise_ph = self.param_noise_actor.action_ph

                        # Configure separate copy for stddev adoption.
                        self.adaptive_param_noise_actor = self.policy(self.sess, self.observation_space,
                                                                      self.action_space, 1, 1, None,
                                                                      **self.policy_kwargs)
                        self.obs_adapt_noise = self.adaptive_param_noise_actor.obs_ph
                        self.action_adapt_noise = self.adaptive_param_noise_actor.action_ph

                    # Inputs.
                    self.obs_train = self.policy_tf.obs_ph
                    self.obs_target = self.target_policy.obs_ph
                    self.action_train_ph = self.policy_tf.action_ph
                    self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
                    self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions')
                    self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
                    self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

                # Create networks and core TF parts that are shared across setup parts.
                self.model_scope = 'model' if not self.is_teacher else '%s_model'%self.name
                with tf.variable_scope(self.model_scope, reuse=False):
                    # CHANGES - adding self.critic_tf_mc and self.critic_with_actor_tf_mc_avg
                    # In plain DDPG don't have the _mc vars, since there are not multiple critic heads
                    if not self.is_teacher:
                        self.actor_tf = self.policy_tf.make_actor(obs0)
                    self.critic_tf, \
                            self.critic_tf_mc = self.policy_tf.make_critic(obs0,
                                                                          self.actions,
                                                                          scope='critic')
                    if not self.is_teacher:
                        self.critic_with_actor_tf, \
                                self.critic_with_actor_tf_mc = self.policy_tf.make_critic(obs0,
                                                                                         self.actor_tf,
                                                                                         scope='critic',
                                                                                         reuse=True)
                        dropout_networks = self.critic_with_actor_tf_mc
                        self.critic_with_actor_tf_mc_avg = tf.reduce_mean(dropout_networks, axis=0)

                # Noise setup
                if self.param_noise is not None and not self.is_teacher:
                    self._setup_param_noise(obs0)

                self.target_scope = 'target' if not self.is_teacher else '%s_target'%self.name
                with tf.variable_scope(self.target_scope, reuse=False):
                    if self.is_teacher or self.use_meta_target:
                        self.target_actor = self.target_policy.make_actor(obs1)
                        critic_target,_ = self.target_policy.make_critic(obs1,
                                                                         self.action_target)
                        self.target_critic = critic_target
                    else:
                        # CHANGES - note '_', just ignore the mc version since not needed
                        critic_target,_ = self.target_policy.make_critic(obs1,
                                                                       self.target_policy.make_actor(obs1))

                summaries = []
                with tf.variable_scope("loss", reuse=False):
                    self.critic_tf = tf.clip_by_value(self.critic_tf, self.return_range[0], self.return_range[1])

                    # CHANGES - note use of self.critic_with_actor_tf_mc_avg
                    # in plain DDPG this is just critic output but as per Henderson et al.
                    # we optimized by avg of critics with dropout instead
                    if not self.is_teacher:
                        self.critic_with_actor_tf = tf.clip_by_value(self.critic_with_actor_tf_mc_avg,
                                             self.return_range[0], self.return_range[1])

                    q_obs1 = critic_target
                    self.target_q = self.rewards + (1. - self.terminals1) * self.gamma * q_obs1

                    append = '_%s'%self.name if self.is_teacher else ''
                    summaries.append(tf.summary.scalar('critic_target'+append, tf.reduce_mean(self.critic_target)))
                    summaries.append(tf.summary.histogram('critic_target'+append, self.critic_target))

                    self._setup_stats()
                    self._setup_target_network_updates()

                if not self.is_teacher:
                    with tf.variable_scope("input_info", reuse=tf.AUTO_REUSE):
                        summaries.append(tf.summary.scalar('rewards', tf.reduce_mean(self.rewards)))
                        summaries.append(tf.summary.histogram('rewards', self.rewards))
                        summaries.append(tf.summary.scalar('param_noise_stddev', tf.reduce_mean(self.param_noise_stddev)))
                        summaries.append(tf.summary.histogram('param_noise_stddev', self.param_noise_stddev))
                        if len(self.observation_space.shape) == 3 and self.observation_space.shape[0] in [1, 3, 4]:
                            summaries.append(tf.summary.image('observation', self.obs_train))
                        else:
                            summaries.append(tf.summary.histogram('observation', self.obs_train))

                with tf.variable_scope("Adam_mpi", reuse=False):
                    if not self.is_teacher:
                        self._setup_actor_optimizer()
                        summaries.append(tf.summary.scalar('actor_loss', self.actor_loss))
                    self._setup_critic_optimizer()
                    summaries.append(tf.summary.scalar('critic_loss'+append, tf.reduce_mean(self.critic_loss)))

                self.params = find_trainable_variables(self.model_scope)
                self.target_params = find_trainable_variables(self.target_scope)

                self.obs_rms_params = [var for var in tf.global_variables()
                                           if "obs_rms" in var.name]
                self.ret_rms_params = [var for var in tf.global_variables()
                                           if "ret_rms" in var.name]

                with self.sess.as_default():
                    self._initialize(self.sess)

                self.summary = tf.summary.merge(summaries)

        logging.info('Finished DDPG model setup')

    def _setup_target_network_updates(self):
        """
        set the target update operations
        """
        init_updates, soft_updates = get_target_updates(tf_util.get_trainable_vars(self.model_scope),
                                                        tf_util.get_trainable_vars(self.target_scope), self.tau, self.verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

    def sample_q_value(self, obs, action=None):
        """
        Samples a q value from one of the bootstrap critic heads. If action is None, evaluates actor's action.
        """
        if action is None:
            to_eval = random.choice(self.critic_with_actor_tf_mc)
            return self.sess.run(to_eval,  {self.obs_train: obs})
        else:
            to_eval = random.choice(self.critic_tf_mc)
            return self.sess.run(to_eval,  {self.obs_train: obs, self.actions: action})

    def q_values(self, obs, action=None):
        """
        Evaluates q values from all of the bootstrap critic heads. If action is None, evaluates actor's action.
        """
        if action is None:
            return self.sess.run(self.critic_with_actor_tf_mc,  {self.obs_train: obs})
        else:
            return self.sess.run(self.critic_tf_mc,  {self.obs_train: obs, self.actions: action})

    def _setup_stats(self):
        """
        setup the running means and std of the inputs and outputs of the model
        """
        ops = []
        names = []

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        if self.include_mc_stats:
            ops += [self.critic_tf_mc]
            names += ['reference_Q_mc']
            ops += [tf.reduce_mean(self.critic_tf_mc, axis=0)]
            names += ['reference_Q_mc_mean']
            ops += [reduce_std(self.critic_tf_mc, axis=0)]
            names += ['reference_Q_mc_std']

        if not self.is_teacher:
            ops += [tf.reduce_mean(self.critic_with_actor_tf)]
            names += ['reference_actor_Q_mean']
            ops += [reduce_std(self.critic_with_actor_tf)]
            names += ['reference_actor_Q_std']

            ops += [tf.reduce_mean(self.actor_tf)]
            names += ['reference_action_mean']
            ops += [reduce_std(self.actor_tf)]
            names += ['reference_action_std']

            if self.param_noise:
                ops += [tf.reduce_mean(self.perturbed_actor_tf)]
                names += ['reference_perturbed_action_mean']
                ops += [reduce_std(self.perturbed_actor_tf)]
                names += ['reference_perturbed_action_std']

        for i in range(len(names)):
            names[i]+='_%s'%self.name

        self.stats_ops = ops
        self.stats_names = names

    def _setup_critic_optimizer(self):
        """
        Setting up critic optimizer, but with additional loss for use of dropout
        """
        if self.verbose >= 2:
            logger.info('setting up critic optimizer')
        critic_target_tf = tf.clip_by_value(self.critic_target, self.return_range[0], self.return_range[1])

        ### eq 10 of the alpha black box dropout
        self.alpha = 0.5
        x = critic_target_tf
        self.flat = self.critic_tf_mc
        flat_stacked = tf.stack(self.flat) # K x M x outsize
        sumsq = tf.reduce_sum(tf.square(x - flat_stacked), axis=-1, keepdims=False) # M x B X outsize
        sumsq *= (-.5 * self.alpha * self.dropout_tau)
        self.critic_loss = (-1.0 * self.alpha ** -1.0) * logsumexp(sumsq, 0)

        # eq 7 from https://arxiv.org/pdf/1506.02142.pdf
        N = self.num_timesteps_final # approx dataset size via constant value
        weight_decay = (self.policy_tf.keep_prob * self.length_scale**2)/(2*N*self.dropout_tau)

        critic_vars = get_vars("%s/critic"%self.model_scope, trainable=True)
        critic_reg_vars = [var for var in critic_vars if \
                           'kernel' in var.name and 'output' not in var.name]
        if self.verbose >= 2:
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with weight decay {}'.format(weight_decay))
        critic_reg = tfc.layers.apply_regularization(
        tfc.layers.l2_regularizer(weight_decay),
            weights_list=critic_reg_vars
        )
        self.critic_loss += critic_reg

        critic_shapes = [var.get_shape().as_list() for var in critic_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, critic_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=critic_vars, beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _initialize(self, sess):
        """
        initialize the model parameters and optimizers
        :param sess: (TensorFlow Session) the current TensorFlow session
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        if not self.is_teacher:
            self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def load(self, load_path):
        data, params = self._load_from_file(load_path)
        restores = []
        for param, loaded_p in zip(self.params + self.target_params, params):
            restores.append(param.assign(loaded_p))
        self.sess.run(restores)
