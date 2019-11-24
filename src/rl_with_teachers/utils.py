
"""log stuff Adapted from OpenAI baselines
A kind of stupid wrapper aroudn python's logging, but some nice TF stuff
"""
import os
import sys
import shutil
import logging
import os.path as osp
import json
import time
import yaml
import numpy as np
import tensorflow as tf
from stable_baselines.common.schedules import Schedule
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise

FORMAT_STR = '%(asctime)s %(process)d %(levelname)-5s @%(filename)s line %(lineno)d func %(funcName)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def configure_file_logging(dir, format_strs=[None], name='log', log_suffix=''):
    try:
        assert isinstance(dir, str)
        os.makedirs(dir)
    except:
        pass

    formatter = logging.Formatter(FORMAT_STR, DATE_FORMAT)
    from logging.handlers import RotatingFileHandler
    Rthandler = RotatingFileHandler("{0}/{1}.log".format(dir, name),
                                     maxBytes=1000*1024*1024,backupCount=10)
    Rthandler.setFormatter(formatter)
    logging.getLogger().addHandler(Rthandler)

def configure_std_logging():
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.INFO)
    if os.isatty(2):
        import coloredlogs
        coloredlogs.install(fmt=FORMAT_STR, level='INFO')
    else:
        formatter = logging.Formatter(FORMAT_STR, DATE_FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

class TensorBoardLogging(object):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """
    def __init__(self, dir):

        try:
            os.makedirs(dir)
        except:
            pass
        self.im_step = 0
        self.dir = dir
        self.writer = tf.summary.FileWriter(dir)

    def write_images(self, image_dict):
        im_summaries = []
        for name, img in image_dict.items():
            step = self.im_step
            if '_' in name:
                parts = name.split('_')
                name = parts[0]
                step = int(parts[1])
            img_sum = tf.summary.image(name=name, tensor=tf.constant(img)).eval()
            self.writer.add_summary(img_sum, step)
        self.im_step+=1
        self.writer.flush()

    def log_histogram(self, tag, values, step, bins):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None

TB_LOGGER = None
def init_logging(log_dir):
    global TB_LOGGER
    path = osp.join(log_dir, 'tb_hists')
    TB_LOGGER = TensorBoardLogging(path)
    configure_std_logging()
    # configure_file_logging(log_dir, format_strs=[None], name='log', log_suffix='')

def log_images(images):
    TB_LOGGER.write_images(images)

def log_histogram(tag, values, step, bins):
    TB_LOGGER.log_histogram(tag, values, step, bins)

### non-log functions
class ExponentialSchedule(Schedule):

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0, decay_rate=0.001):
        """
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        decay_rate: float
            rate to decay at
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.decay_rate = decay_rate

    def value(self, t):
        """See Schedule.value"""

def save_experiences(filename, memory):
    """
    Save experiences of agents as pickle file
    """
    f = open(filename, "wb")
    memo_array = [buff.get_batch(np.arange(buff.length))
        for buff in [memory.observations0, memory.actions, memory.rewards, memory.observations1, memory.terminals1]]

    memo_array2 = []
    for (o0, a, r, o1, d) in zip(*memo_array):
        rr = np.asscalar(r)
        dd = d[0]==1
        memo_array2 += ((o0, a, rr, o1, dd),)

    pickle.dump(memo_array2, f)
    print("Saved experiences to file")
    f.close()

def parse_noise_types(noise_type, nb_actions):
    """
    Parse noise types for policies
    """
    action_noise = None
    param_noise = None
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    return action_noise, param_noise

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def parse_conf(config_paths, args):
    """
    Parse conf to run with from yaml files and command line args.
    Any values set as command line args supercede files ones.
    """
    conf = {}
    for config_path in config_paths.split(','):
        with open(config_path, 'r') as conf_file:
            this_conf = yaml.load(conf_file)
            dict_merge(conf, this_conf)

        while 'include' in conf:
            for conf_include_f in conf.pop('include'):
                with open(conf_include_f, 'r') as conf_file:
                    conf_include = yaml.load(conf_file)
                    for key in conf_include:
                        # Newer confs overwrite older ones
                        if key=='include':
                            conf[key]+= conf_include[key]
                        elif key not in this_conf:
                            conf[key] = conf_include[key]

    for key in args:
        if args[key] is not None:
            conf[key] = args[key]
    return conf
