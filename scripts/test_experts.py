"""
Utility script to visualize different experts on the mujoco points fetch env.
"""

import random
import argparse
import gym
from PIL import Image
from rl_with_teachers.envs import *

def run_normal_rollout(env, teacher=None):
    ob = env.reset()
    done = False
    num_steps = 0
    ret = 0
    while not done:
        print(num_steps)
        print("ob: {}".format(ob))
        if teacher is not None:
            ac = teacher(ob)
        else:
            ac = env.action_space.sample()
        ob, r, done, _ = env.step(ac)
        print("ac: {}".format(ac))
        print("r: {}\n".format(r))
        ret += r
        num_steps += 1
        env.render()
    print("ob: {}".format(ob))
    print("Return {} in {} steps.".format(ret, num_steps))

def run_adversarial_rollout(env, teacher, bad_teacher):
    ob = env.reset()
    done = False
    num_steps = 0
    ret = 0
    while not done:
        print(num_steps)
        print("ob: {}".format(ob))
        if num_steps % 3 == 0:
            ac = bad_teacher(ob)
        else:
            ac = teacher(ob)
        ob, r, done, _ = env.step(ac)
        print("ac: {}".format(ac))
        print("r: {}\n".format(r))
        ret += r
        num_steps += 1
        env.render()
        # print(ob)
    print("ob: {}".format(ob))
    print("Return {} in {} steps.".format(ret, num_steps))

def run_midpoint_rollout(env, teacher_1, teacher_2, steps=20):
    ob = env.reset()
    done = False
    num_steps = 0
    ret = 0
    for _ in range(steps):
        print(num_steps)
        print("ob: {}".format(ob))
        ac = teacher_1(ob)
        ob, r, done, _ = env.step(ac)
        print("ac: {}".format(ac))
        print("r: {}\n".format(r))
        ret += r
        num_steps += 1
        env.render()
    for _ in range(steps):
        print(num_steps)
        print("ob: {}".format(ob))
        ac = teacher_2(ob)
        ob, r, done, _ = env.step(ac)
        print("ac: {}".format(ac))
        print("r: {}\n".format(r))
        ret += r
        num_steps += 1
        env.render()
    print("ob: {}".format(ob))
    print("Return {} in {} steps.".format(ret, num_steps))

def run_sequenced_rollout(env, teachers, steps=20):
    ob = env.reset()
    done = False
    num_steps = 0
    ret = 0
    # A = env.sim.render(500, 500, camera_name='external_camera_0')[::-1]
    # A = env.render(mode='rgb_array')
    # save_image(A, path="./tmp/{}.png".format(num_steps))
    for i, teacher in enumerate(teachers):
        if isinstance(steps, list):
            steps_i = steps[i]
        else:
            steps_i = steps
        for _ in range(steps_i):
            print(num_steps)
            print("ob: {}".format(ob))
            ac = teacher(ob)
            ob, r, done, _ = env.step(ac)
            print("ac: {}".format(ac))
            print("r: {}\n".format(r))
            ret += r
            num_steps += 1
            env.render()
            # A = env.render(mode='rgb_array')
            # A = env.sim.render(500, 500, camera_name='external_camera_0')[::-1]
            # save_image(A, path="./tmp/{}.png".format(num_steps))
    print("ob: {}".format(ob))
    print("Return {} in {} steps.".format(ret, num_steps))

def save_image(arr, path):
    """
    Save a numpy array representing an image to disk.
    :param arr: numpy array corresponding to image
    :param path: string corresponding to the path to save it at
    """
    im = Image.fromarray(arr)
    im.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="SparseGoalInStatePointsPath-v0",
    )
    parser.add_argument(
        "--type",
        type=str,
    )
    parser.add_argument(
        "--render_video", 
        action='store_true',
    )
    args = parser.parse_args()

    assert(args.type in ["random", "full", "partial", "adversarial", "midpoint"])
    assert(not args.render_video)

    ### TODO: decide if we want an offscreen render option to dump rollout videos or images ###
    ### TODO: use the type argument in a better, cleaner way to construct teachers? ###
    ### TODO: check environment names... ###
    ### TODO: fix the HookSweep partial case in a clean way, it will only work 25% of the time right now lol ###

    env = gym.make(args.env)
    env.reset()

    if args.type == "random":
        run_normal_rollout(env, teacher=None)
        exit(0)

    if args.env == "SparseGoalInStatePointsPath-v0":

        if args.type == "full":
            teacher = OptimalPathAgent(env, adversarial=False)
            run_normal_rollout(env, teacher=teacher)

        elif args.type == "partial":
            teachers = [OneGoalPathAgent(env, env.path_points[p], adversarial=False) for p in range(len(env.path_points))]
            run_sequenced_rollout(env, teachers=teachers, steps=20)

        elif args.type == "adversarial":
            teacher = OptimalPathAgent(env, adversarial=False)
            bad_teacher = OptimalPathAgent(env, adversarial=True)
            run_adversarial_rollout(env, teacher=teacher, bad_teacher=bad_teacher)

        else:
            midpoint_teacher = OptimalPathHalfwayAgent(env)
            switch_teacher = OptimalPathSwitchAgent(env)
            teachers = [midpoint_teacher, switch_teacher]
            teachers = teachers * 4
            run_sequenced_rollout(env, teachers=teachers, steps=20)

    elif args.env == "OneGoalPickPlaceEnv-v0":

        if args.type == "full":
            GOAL = np.array([1.45, 0.55, 0.425])
            teacher = OptimalPickPlaceAgent(goal=GOAL)
            run_normal_rollout(env, teacher=teacher)

        elif args.type == "partial":
            GOAL = np.array([1.45, 0.55, 0.425])
            picker = PickAgent(return_to_start=True)
            placer = PlaceAgent(GOAL)
            teachers = [picker, placer]
            run_sequenced_rollout(env, teachers=teachers, steps=[30, 30])

        else:
            raise Exception("Invalid teacher type {} for env {}".format(args.type, args.env))

    elif args.env == "FetchHookSweepPushBetterDenseEasyInitEasierTaskCloseInitNoGrasp-v0":

        if args.type == "full":
            teacher = FullHookSweepPushAgent(short_hook=env.unwrapped.short_hook, naive=False, cube_side_init=env.unwrapped.cube_side_init)
            run_normal_rollout(env, teacher=teacher)

        elif args.type == "partial":
            pick_hooker = PickHookAgent()
            position_hooker = PositionForSweepPushHookAgent(short_hook=env.unwrapped.short_hook)
            sweep_hooker = SweepHookAgent()
            push_hooker = PushHookAgent(short_hook=env.unwrapped.short_hook)

            # this will work 25% of the time since we don't know which instance we're in lol
            if random.random() < 0.5:
                teachers = [pick_hooker, position_hooker, sweep_hooker]
            else:
                teachers = [pick_hooker, position_hooker, push_hooker]
            run_sequenced_rollout(env, teachers=teachers, steps=[10, 20, 20])

        else:
            raise Exception("Invalid teacher type {} for env {}".format(args.type, args.env))

    else:
        raise Exception("Got invalid environment: {}".format(args.env))
