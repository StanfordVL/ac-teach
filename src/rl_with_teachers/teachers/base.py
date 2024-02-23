import abc # for abstract base class definitions
import six # preserve metaclass compatibility between python 2 and 3
import numpy as np

@six.add_metaclass(abc.ABCMeta)
class TeacherPolicy:
    """
    Abstract base class for all teacher policies.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, obs):
        """
        Takes a single observation and returns
        a single action.
        """
        raise NotImplementedError

    def is_callable(self, obs):
        return True

    def reset(self):
        """
        Reset any internal state on end of episode.
        """
        pass

    def apply_noise(self, action):
        if self.noise is not None:
            action+=self.noise()
            if np.max(np.abs(action))>1.0:
                action=action/np.max(np.abs(action))

        return action

class RandomAgent(TeacherPolicy):
    """
    An agent that returns an action uniformly at random.
    """
    def __init__(self, env):
        self.env = env

    def is_callable(self, obs):
        return True

    def __call__(self, obs):
        return self.env.action_space.sample()


class IRISAgent(TeacherPolicy):
    """
    An expert based off of the hierarchical IRIS policy.
    """
    def __init__(self, env, agent, noise=None):
        self.noise = noise

        import batchRL
        from batchRL.config import IRISConfig
        from batchRL.algo import IRISAlgo
        self.iris_config = IRISConfig()
        self.iris_config.experiment.env = env
        self.iris_algo = IRISAlgo(self.iris_config)
        self.iris_algo._prepare_for_test_rollouts(
            agent_path=agent,
            seeding=False, 
            delete_env=True,
        )
        self.iris_algo.reset_external()

    def __call__(self, obs):
        action = self.iris_algo.external_act(obs)
        action = self.apply_noise(action)
        return action

    def reset(self):
        self.iris_algo.reset_external()

    def apply_noise(self, action):
        # ignore noise for now
        return action

        # if self.noise is not None:
        #     action + =self.noise()
        #     if np.max(np.abs(action)) > 1.0:
        #         action = action / np.max(np.abs(action))
        # return action
