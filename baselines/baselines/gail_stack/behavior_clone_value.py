'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf

from baselines.gail import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail_stack.run_jaco1 import runner
from baselines.gail_stack.dataset.mujoco_dset_value import Mujoco_Dset

from jaco_arm import JacoStackEnv as JacoEnv
import numpy as np

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/stack_rew.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e3)
    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    ret = tf.placeholder(dtype=tf.float32, shape=[None])

    stochastic = U.get_placeholder_cached(name="stochastic")
    policy_loss = tf.reduce_mean(tf.square(ac-pi.ac))
    value_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    all_var_list = pi.get_trainable_variables()
    policy_var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    value_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    assert len(policy_var_list) == len(value_var_list) + 1
    policy_adam = MpiAdam(policy_var_list, epsilon=adam_epsilon)
    value_adam = MpiAdam(value_var_list, epsilon=adam_epsilon)
    policy_lossandgrad = U.function([ob, ac, stochastic], [policy_loss]+[U.flatgrad(policy_loss, policy_var_list)])
    value_lossandgrad = U.function([ob, ret, stochastic], U.flatgrad(value_loss, value_var_list))

    U.initialize()
    policy_adam.sync()
    value_adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert, rets = dataset.get_next_batch(optim_batch_size, 'train')
        policy_train_loss, policy_g = policy_lossandgrad(ob_expert, ac_expert, True)
        value_g = value_lossandgrad(ob_expert, rets, True)
        #policy_adam.update(policy_g, optim_stepsize)
        value_adam.update(value_g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert, ret = dataset.get_next_batch(-1, 'val')
            policy_val_loss, _ = policy_lossandgrad(ob_expert, ac_expert, True)
            print('######################pl', policy_val_loss)
            logger.log("[Policy] Training loss: {}, Validation loss: {}".format(policy_train_loss, policy_val_loss))

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    #U.save_state(savedir_fname, var_list=pi.get_variables())
    U.save_variables(savedir_fname, variables=pi.get_variables())
    return savedir_fname


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = JacoEnv()#env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    #env = bench.Monitor(env, logger.get_dir() and
    #                    osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    savedir_fname = learn(env,
                          policy_fn,
                          dataset,
                          max_iters=args.BC_max_iter,
                          ckpt_dir=args.checkpoint_dir,
                          log_dir=args.log_dir,
                          task_name=task_name,
                          verbose=True)
    """
    avg_len, avg_ret = runner(env,
                              policy_fn,
                              savedir_fname,
                              timesteps_per_batch=1024,
                              number_trajs=10,
                              stochastic_policy=args.stochastic_policy,
                              save=args.save_sample,
                              reuse=True)
    """

if __name__ == '__main__':
    args = argsparser()
    main(args)
