# AC-Teach

Code for the CoRL 2019 paper [AC-Teach: A Bayesian Actor-Critic Method for Policy Learning with an Ensemble of Suboptimal Teachers](https://sites.google.com/view/acteach/)

## Installation

Requires a [MuJoCo license](https://www.roboti.us/license.html) and must have mujoco installed.
Please see instructions [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key) to install MuJoCo.

Optionally create a new virtual environment and activate it with:
```
virtualenv -p python3 env
source env/bin/activate
```

Then, install the repository and dependencies with:
```
pip install -r requirements.txt
pip install -e .
```

## Training

Here's an example to train a bayesian DDPG policy with the `AC-Teach` behavioral policy on the `pick-place` task using both a `pick` and `place` teacher with this command:
```
python scripts/run.py --config cfg/pick_place/experiments/efficiency/partial_complete_suboptimal/train_ours.yaml
```

You can optionally specify the location of log files with the following arguments, otherwise a `logs` directory is created under the base repository folder.
```
--log-base-dir : base directory to use for logs (a log directory will automatically be created using the environment and experiment name)
--experiment-name : name used in creating log files
--log-dir : full path to a desired log directory (overrides the above two arguments)
```

To view tensorboard logs, navigate to the log directory and run
```
tensorboard --logdir .
```

## Evaluation

To evaluate and visualize a trained agent, run the following, replacing the `--config` argument with the appropriate eval config for the environment, and the `--load-path` argument with the agent checkpoint:
```
python scripts/run.py --config cfg/pick_place/eval.yaml --load-path /path/to/checkpoint
```
To avoid rendering to the screen, pass `--render-eval 0`.

A jupyter notebook, 'Analyze Logs.ipynb', is also included under the scripts directory for processings sets of logs for plots with variance and multiple policies. Note that it assumes it is in the base log directory.

## Adding a new environment or new teachers

You can copy the structure of the configs from another environment into `cfg/your_environment` and make sure the environment is registered with Gym (see [here](https://github.com/StanfordVL/ac-teach/blob/master/src/rl_with_teachers/envs/pick_place.py#L101) for example).

Then, you should add a file under `src/ac-teach/teachers/` (see [here](https://github.com/StanfordVL/ac-teach/blob/master/src/rl_with_teachers/teachers/pick_place.py) for an example).

Finally, ensure your environment has a `make_teachers` function (see [here](https://github.com/StanfordVL/ac-teach/blob/master/src/rl_with_teachers/envs/pick_place.py#L86) for an example) that associates the `type` string argument with a set of teachers. This argument can be specified in `cfg/your_environment/teachers/your_teacher_config.yaml` ([here](https://github.com/StanfordVL/ac-teach/blob/master/cfg/pick_place/teachers/partial_complete_suboptimal.yaml) is an example), and you can link this teacher configuration to your main configuration ([example](https://github.com/StanfordVL/ac-teach/blob/master/cfg/pick_place/experiments/efficiency/partial_complete_suboptimal/train_ours.yaml#L3)).


## Experimenting with AC-Teach algorithm
The AC-Teach algorithm is implemented in the ACTeachStrategy in behavior_policies.py. You can vary parameters to it in the yaml files, or implement your own behavior policies and add them to make_behavior_policy in behavior_policies.py to compare them to AC-Teach.

## Citation

If you use this code in your work please cite our paper:

```
@article{kurenkov2019ac,
  title={AC-Teach: A Bayesian Actor-Critic Method for Policy Learning with an Ensemble of Suboptimal Teachers},
  author={Kurenkov, Andrey and Mandlekar, Ajay and Martin-Martin, Roberto and Savarese, Silvio and Garg, Animesh},
  journal={arXiv preprint arXiv:1909.04121},
  year={2019}
}
```
