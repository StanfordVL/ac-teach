import numpy as np
import math
import random
import logging
from abc import ABC, abstractmethod
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.common.schedules import LinearSchedule
from rl_with_teachers.utils import parse_noise_types

def make_behavior_policy(env_id, base_env, learner, teach_behavior_policy, teachers_conf, behavior_policy_params):
    """
    Create a behavioral policy for training.

    :param env_id: The name of the id for training
    :param base_env: An actual instance of the env, without any wrappers around it
    :param learner: The learner object (one of the classes from learners.py)
    :param teach_behavior_policy: The name of the behavior_policy to create
    :param teachers_conf: The configuration for the teachers
    :param behavior_policy_params: The params for the behavior policy
    """
    teachers = []
    behavior_policy = None
    if teach_behavior_policy is None:
        return None

    for teacher_conf in teachers_conf:
        action_noise = None
        if 'noise_type' in teacher_conf:
            nb_actions = base_env.action_space.shape[-1]
            noise, _ = parse_noise_types(teacher_conf.pop('noise_type'), nb_actions)
        else:
            noise = None
        if teacher_conf['type'] == 'pretrained':
            pretrained_agent = PPO2.load(teacher_conf['model_path']%env_id)
            if noise is not None:
                teacher_fns = [lambda obs: pretrained_agent.predict(obs, deterministic=False)[0]/np.abs(learner.action_space.low) + noise()]
            else:
                teacher_fns = [lambda obs: pretrained_agent.predict(obs, deterministic=False)[0]/np.abs(learner.action_space.low)]
        else:
            teacher_fns = base_env.make_teachers(teacher_conf.pop('type'), noise, **teacher_conf)
        teachers+=teacher_fns

    if teach_behavior_policy == 'random':
        behavior_policy = RandomBehaviorPolicy(base_env, learner, teachers, **behavior_policy_params)
    elif teach_behavior_policy == 'only_teacher':
        behavior_policy = OnlyTeacherBehaviorPolicy(base_env, learner, teachers)
    elif teach_behavior_policy == 'optimal':
        if env_id == 'OneGoalPickPlaceEnv-v0':
            behavior_policy = OptimalPickPlaceBehaviorPolicy(learner, teachers)
        elif env_id == 'SanityPath-v0':
            behavior_policy = OptimalPathBehaviorPolicy(learner, teachers)
        elif env_id == 'SanityPathMujoco-v0':
            behavior_policy = OptimalReachBehaviorPolicy(learner, teachers)

    elif teach_behavior_policy == 'dqn' or teach_behavior_policy == 'bdqn':
        dqn_env = gym.make(env_id)
        if 'use_learner' not in behavior_policy_params or behavior_policy_params['use_learner']:
            dqn_env.action_space = spaces.Discrete(len(teachers)+1)
        else:
            dqn_env.action_space = spaces.Discrete(len(teachers))
        if teach_behavior_policy == 'dqn':
            dqn = NiceDQN(
                    env=dqn_env,
                    policy=MlpPolicy,
                    **behavior_policy_params.pop('dqn_params')
            )

            dqn.exploration = LinearSchedule(schedule_timesteps=int(behavior_policy_params.pop('num_timesteps')*0.2),
                                             initial_p=1.0,
                                             final_p=0.02)
            behavior_policy = DQNChoiceBehaviorPolicy(base_env, learner, teachers, dqn, **behavior_policy_params)
        else:
            bdqn = BDQN(
                    env=dqn_env,
                    **behavior_policy_params.pop('bdqn_params')
            )
            behavior_policy = BDQNUncertaintyBehaviorPolicy(base_env, learner, teachers, bdqn, **behavior_policy_params)
    elif teach_behavior_policy == 'critic':
            behavior_policy = CriticBehaviorPolicy(base_env, learner, teachers, learner, **behavior_policy_params)
    elif teach_behavior_policy == 'acteach':
            behavior_policy = ACTeachBehaviorPolicy(base_env, learner, teachers, learner, **behavior_policy_params)
    elif teach_behavior_policy is not None:
            raise ValueError('%s not a valid learning behavior policy'%teach_behavior_policy)

    return behavior_policy

class RLwTeachersBehaviorPolicy(ABC):
    """ An abstract class to encapsulate different ways of doing RL w Teachers

    :param env: A gym env object
    :param learner: The learner object
    :param teachers: (list) list of teacher objects ; should be callable with one param, obs, to get action
    :param use_learner: (bool) Whether learner should be counted as one of the policies to use or not
    """

    def __init__(self, env, learner, teachers, use_learner=True):
        self.env = env
        self.learner = learner
        self.teachers = teachers
        self.num_policies = len(teachers) + 1
        self.use_learner = use_learner

    def get_all_actions(self, obs):
        learner_action, q = self.learner.get_action(obs, compute_q=True)
        actions = [teacher(obs) for teacher in self.teachers]
        return [learner_action] + actions

    def get_action(self, index, obs, with_exploration=False):
        if not self.use_learner:
            index+=1 #account for zero being agent
        if index == -1:
            return self.env.action_space.sample()
        elif index == 0:
            return self.learner.select_action(obs, with_exploration=with_exploration)
        else:
            return self.teachers[index-1](obs)

    def get_actions(self, indeces, obs):
        actions = []
        for i,idx in enumerate(indeces):
            action = self.get_action(idx,obs[i])
            actions.append(action)
        return np.array(actions)

    def reset(self):
        """ Reset on  episode reset

        By default does nothing
        """
        for teacher in self.teachers:
            teacher.reset()

    @abstractmethod
    def choose_action(self, obs):
        """ Given observation, choose among actions to take
        Should return the policy index, action, and q values
        """
        pass

    def step_learn(self, obs, policy_choice, reward, new_obs, done):
        """ Given memory of past transitions, train the selection behavior_policy.
        Return TF summary to log.
        By default does nothing and returns None.
        """
        return None

class OnlyTeacherBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    A behavioral policy that is used with only one teacher, and always calls it and not
    the learning agent.
    """

    def __init__(self,
                 env,
                 learner,
                 teachers):
        super(OnlyTeacherBehaviorPolicy, self).__init__(env, learner, teachers, True)

    def choose_action(self, obs):
        """ Always just picks 1 for optimal behavior_policy """
        policy_choice = 1
        return policy_choice, self.get_action(policy_choice, obs, with_exploration=True), None

class RandomBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    A behavioral policy that just chooses policies at random.
    Can optionally configure to have non uniform probabilities or commit to the same choice
    for a sequence of steps.

    :param env: A gym env object
    :param learner: The learner object
    :param teachers: (list) list of teacher objects ; should be callable with one param, obs, to get action
    :param use_learner: (bool) Whether learner should be counted as one of the policies to use or not
    :param probs: (list) optional list of probabilities to use for sampling poicy choice at each time step
    :param commit_per_episode: (bool) Whether to just make a choice at start of episode and stick with until next episode
    :param reuse_prob: (float) if commiting per episode, this is probability to call on chosen policy or at random
    :param reuse_prob_decay: (float) if commiting per episode, this is a multiplicative factor to apply to reuse_prob at each step
    """

    def __init__(self,
                 env,
                 learner,
                 teachers,
                 use_learner=True,
                 probs=None,
                 commit_per_episode=False,
                 reuse_prob=1.0,
                 reuse_prob_decay=1.0):
        super(RandomBehaviorPolicy, self).__init__(env, learner, teachers, use_learner)
        self.probs = probs
        self.commit_per_episode = commit_per_episode
        self.current_policy = None
        self.reuse_prob = reuse_prob
        self.reuse_prob_decay = reuse_prob_decay
        self.current_reuse_prob = self.reuse_prob

    def choose_action(self, obs):
        """ Given observation, choose among actions to take
        Should return the policy index, action, and q value
        """
        if self.commit_per_episode and self.current_policy is not None:
            policy_choice = self.current_policy
            if random.random() > self.current_reuse_prob: # with prob 1-self.current_reuse_prob act random, else follow teacher
                policy_choice = -1
            self.current_reuse_prob*=self.reuse_prob_decay # decrease reuse probability
        else:
            if self.probs is None:
                policy_choice = random.randint(0,self.num_policies-1)
            else:
                policy_choice = np.random.choice(np.arange(self.num_policies), p=self.probs)
            self.current_policy = policy_choice
        return policy_choice, self.get_action(policy_choice, obs, with_exploration=True), None

    def reset(self):
        """ Optional learn method """
        self.current_policy = None
        self.current_reuse_prob = self.reuse_prob

class OptimalReachBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    Hardcoded optimal reach behavioral policy.
    Environment specific because the logic for optimally choosing among
    the env's teachers is itself environment specific.
    """

    def choose_action(self, obs):
        """ Given observation, choose among actions to take
        Should return the policy index, action, and q value
        """
        goal = FetchOneGoalReachEnv.GOAL
        pos = obs[:3]
        if np.linalg.norm(pos - goal) > np.linalg.norm(pos - START_POINT):
            index = 1
        else:
            if np.linalg.norm(pos - goal) < np.linalg.norm(goal - START_POINT)*0.2:
                index = 0
            else:
                index = 2
        return index, self.get_action(index, obs), 0.0

class OptimalPathBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    Hardcoded optimal path behavioral policy.
    Environment specific because the logic for optimally choosing among
    the env's teachers is itself environment specific.
    """

    def choose_action(self, obs):
        """ Given observation, choose among actions to take
        Should return the policy index, action, and q value
        """
        # Assume 2D case
        index = 1
        for i in range(2,len(obs)):
            if obs[i]:
                index+=1
        return index, self.get_action(index, obs), 0.0

class OptimalPickPlaceBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    Hardcoded Pick Place reach behavioral policy.
    Environment specific because the logic for optimally choosing among
    the env's teachers is itself environment specific.
    """

    def choose_action(self, obs):
        """ Given observation, choose among actions to take
        Should return the policy index, action, and q value
        """
        if not self.teachers[0].has_grasped:
            index = 1
        else:
            index = 2
        return index, self.get_action(index, obs), 0.0

class DQNChoiceBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    A behavioral policy that chooses the policy by training a DQN to make the choice.

    :param env: A gym env object
    :param learner: The learner object
    :param teachers: (list) list of teacher objects ; should be callable with one param, obs, to get action
    :param dqn: The DQN object to do choices and learning with
    :param commit_time: (int) optionally, commit to same policy for a number of steps after each choice
    :param do_exploration: (bool) whether to do epsilon exploration when making choice with DQN
    :param use_learner: (bool) Whether learner should be counted as one of the policies to use or not
    """

    def __init__(self, env, learner, teachers, dqn, commit_time = 0, do_exploration=True, use_learner=True):
        super(DQNChoiceBehaviorPolicy, self).__init__(env, learner, teachers, use_learner=use_learner)
        self.num_train_steps = 0
        self.num_choice_steps = 0
        self.dqn = dqn
        self.commit_time = commit_time
        self.do_exploration = do_exploration
        self.current_commit_time = 0
        self.current_choice = None

    def choose_action(self, obs):
        with self.dqn.sess.as_default():
            _, q_values, _ = self.dqn.step_model.step(obs[None], deterministic = True)
            if self.current_commit_time > 0:
                agent_choice = self.current_choice
                self.current_commit_time-=1
            elif self.current_commit_time==-1 and self.current_choice and \
                    self.teachers[self.current_choice-1].is_callable(obs):
                agent_choice = self.current_choice
            else:
                if self.do_exploration:
                    update_eps = self.dqn.exploration.value(self.num_choice_steps)
                    agent_choice = self.dqn.act(np.array(obs)[None], update_eps=update_eps)[0]
                else:
                    agent_choice = np.argmax(q_values[0])
                self.current_choice = agent_choice
                self.current_commit_time = self.commit_time
            q_val = q_values[0][agent_choice]
            self.num_choice_steps+=1
            return agent_choice, self.get_action(agent_choice, obs, with_exploration=True), q_val

    def choose_action_greedy(self, obs):
        with self.dqn.sess.as_default():
            _, q_values, _ = self.dqn.step_model.step(obs[None], deterministic = True)
            agent_choices = np.argmax(q_values[0])
            return agent_choices, self.get_action(agent_choices, obs), q_values[0,agent_choices]

    def reset(self):
        """ Reset on  episode reset

        By default does nothing
        """
        super().reset()
        self.current_commit_time = 0
        self.current_choice = None

    def step_learn(self, obs, policy_choice, reward, new_obs, done):
        """ Given memory of past transitions, train the selection behavior_policy.
        Return TF summary to log.
        By default does nothing and returns None.
        """
        self.dqn.store_and_train(obs, policy_choice, reward, new_obs, done)

class CriticBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    A behavioral policy that chooses the policy by using the DDPG critic.

    :param env: A gym env object
    :param learner: The learner object
    :param teachers: (list) list of teacher objects ; should be callable with one param, obs, to get action
    :param critic_model: The critic object from DDPG to use for choosing policies to follow
    :param use_learner: (bool) Whether learner should be counted as one of the policies to use or not
    """
    def __init__(self, env, learner, teachers, critic_model, use_learner=True):
        super(CriticBehaviorPolicy, self).__init__(env, learner, teachers, use_learner=use_learner)
        self.num_choice_steps = 0
        self.critic_model = critic_model
        self.current_policy_choice = None

    def choose_action(self, obs):
        learner_action = self.get_action(0, obs, with_exploration=True)
        learner_q = self.critic_model.eval_q_value(obs[None], learner_action[None])
        qs = [learner_q]
        actions = [learner_action]
        for teacher in self.teachers:
            action = teacher(obs)
            qs.append(self.critic_model.eval_q_value(obs[None], action[None]))
            actions.append(action)
        logging.info('Critic evaluation of policy actions: %s'%str(np.squeeze(qs)))

        agent_choice = np.argmax(qs)
        self.num_choice_steps+=1

        return agent_choice, actions[agent_choice], qs[agent_choice]

class ACTeachBehaviorPolicy(RLwTeachersBehaviorPolicy):
    """
    A behavioral policy that chooses the policy by using the bayesian DDPG critic, with an added commitment mechanism)

    :param env: A gym env object
    :param learner: The learner object
    :param teachers: (list) list of teacher objects ; should be callable with one param, obs, to get action
    :param uncertainty_model: The bayesian critic object from DDPG to use for choosing policies to follow
    :param use_learner: (bool) Whether learner should be counted as one of the policies to use or not
    :param commit_thresh: (float) The threshold of confidence new policy choice is better than prior one when choosing to switch
    :param with_commitment: (bool) Whether to do commitment, or just choose new policy at each step
    :param decay_commitment: (bool) Whether to decrease the commit_thresh at each time step, if doing commitment
    :param commitment_decay_coeff: (float) if doing commitment with decay, the factor to multiply commit threshold by (should be less than 1.0)
    """
    def __init__(self, env, learner, teachers, uncertainty_model,
                       use_learner=True,
                       commitment_thresh=0.6, with_commitment = True,
                       decay_commitment=True, commitment_decay_coeff = 0.99):
        super(ACTeachBehaviorPolicy, self).__init__(env, learner, teachers, use_learner=use_learner)
        self.num_choice_steps = 0
        self.uncertainty_model = uncertainty_model
        self.with_commitment = with_commitment
        self.current_policy_choice = None
        self.commitment_thresh = commitment_thresh
        self.num_steps_commited = 0
        self.decay_commitment = decay_commitment
        self.commitment_decay_coeff = commitment_decay_coeff

    def choose_action(self, obs):
        learner_action = self.get_action(0, obs, with_exploration=True)
        learner_q = self.uncertainty_model.sample_q_value(obs[None], learner_action[None])
        qs = [learner_q]
        actions = [learner_action]
        for teacher in self.teachers:
            action = teacher(obs)
            # Thompson sampling of Q value for each action
            qs.append(self.uncertainty_model.sample_q_value(obs[None], action[None]))
            actions.append(action)

        logging.info('ACT choosing action for state %s'%str(np.squeeze(obs)))
        logging.info('Policy actions at state are %s'%str(actions))
        logging.info('Critic evaluation of policy actions is %s'%str(np.squeeze(qs)))

        max_q_choice = np.argmax(qs)
        if self.current_policy_choice is None:
            self.current_policy_choice = max_q_choice

        # If using commitment and current max choice is not same as last time, need to choose whether to switch
        if self.with_commitment and self.current_policy_choice != max_q_choice:
            current_qs = self.uncertainty_model.q_values(obs[None], actions[self.current_policy_choice][None])
            new_qs = self.uncertainty_model.q_values(obs[None], actions[max_q_choice][None])
            current_mean = np.mean(current_qs)
            new_mean = np.mean(new_qs)
            current_var = np.var(current_qs)
            new_var = np.var(new_qs)
            diff = np.array(new_qs) - np.array(current_qs)
            diff_mean = current_mean - new_mean
            diff_var = current_var + new_var + 1.0/self.uncertainty_model.dropout_tau

            # Calculation of probability Q of new policy is in fact better than the policy last used
            prob_greater = 1.0 - (1.0 + math.erf((0.0 - diff_mean + 0.000000000001)/ math.sqrt(diff_var*2.0))) / 2.0

            logging.info('Considering switch %d->%d'%(self.current_policy_choice, max_q_choice))
            logging.info('Mean and var of policy %d: %f %f'%(self.current_policy_choice, current_mean, current_var))
            logging.info('Mean and var of policy %d: %f %f'%(max_q_choice, new_mean, new_var))
            logging.info('Diff mean %f and var %f, probability worth it is %f'%(diff_mean, diff_var, prob_greater))

            commit_thresh = self.commitment_thresh
            if self.decay_commitment:
                commit_thresh*= self.commitment_decay_coeff**self.num_steps_commited
            logging.info('Num steps same policy %d, threshold at %f'%(self.num_steps_commited, commit_thresh))

            if prob_greater > commit_thresh: # If confident enough, make the switch
                agent_choice = max_q_choice
                self.num_steps_commited = 0
            else: # Else, stay with same policy as last time
                agent_choice = self.current_policy_choice
                self.num_steps_commited += 1
        else: # If no commitment, just always go with policy having highest Q value evaluation
            agent_choice = max_q_choice

        self.current_policy_choice = agent_choice
        self.num_choice_steps+=1

        return agent_choice, actions[agent_choice], qs[agent_choice]

    def choose_actions_greedy(self, obs):
        learner_actions = self.uncertainty_model.eval_actor(obs)
        learner_qs = self.uncertainty_model.eval_q_value(obs)
        all_actions = learner_actions[:, np.newaxis, :]
        all_qs = [learner_qs]
        for teacher in self.teachers:
            actions = np.concatenate([teacher(obs[i])[None] for i in range(obs.shape[0])], axis=0)
            all_actions = np.concatenate([all_actions,actions[:, np.newaxis, :]], axis=1)
            teacher_qs = self.uncertainty_model.eval_q_value(obs, actions)
            all_qs.append(teacher_qs)
        all_qs = np.squeeze(np.stack(all_qs, axis=-1))
        agent_choices = np.squeeze(np.argmax(all_qs, axis=-1))
        return agent_choices, all_actions[np.arange(len(agent_choices)), tuple(agent_choices)], all_qs[:,agent_choices]

    def reset(self):
        """ Reset on  episode reset

        By default does nothing
        """
        super().reset()
        self.current_policy_choice = None
        self.num_steps_commited = 0
