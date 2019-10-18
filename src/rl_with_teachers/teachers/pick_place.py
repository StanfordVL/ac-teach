from rl_with_teachers.teachers.base import TeacherPolicy
import numpy as np

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

        noisy_action = self.apply_noise(np.array(action))
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
        noisy_action = self.apply_noise(np.array(action))
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

        noisy_action = self.apply_noise(np.array(action))
        noisy_action[-1] = action[-1]#0.95*action[-1]+0.05*noisy_action[-1]
        return noisy_action
