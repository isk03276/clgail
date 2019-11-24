'''
Disclaimer: The trpo part highly rely on trpo_mpi at @openai/baselines
'''

import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.gail.statistics import stats
import threading

from jaco_arm import JacoEnv
from baselines.agail_stack.adversary import TransitionClassifier
from baselines.agail_stack import mlp_policy
import mujoco_py
import copy

import random

pi = None
reward_giver = None
worker_pi_0 = None
worker_pi_1 = None
worker_pi_2 = None
worker_disc_0 = None
worker_disc_1 = None
worker_disc_2 = None

lock = threading.Lock()
#lock = False

test  = 0

def evaluate(pi, viewer, sr_file_name, env, reward_giver, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            with open(sr_file_name, "a") as f:
                f.write(str(sum(ep_true_rets)/len(ep_true_rets)) + '\n')
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            break
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_giver.get_reward(ob, ac)
        ob, true_rew, new, _ = env.step(ac)

        #viewer.render()

        rews[i] = rew + true_rew# + sub_goal_rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

class Worker(threading.Thread):
    def __init__(self, worker_id, global_update_func, env, policy_net_hid_size, entcoeff, horizon, sr_file_name, initial_states, stochastic=True):
        #U.make_session(num_cpu=1, graph=tf.get_default_graph()).__enter__()
        #U.make_session(num_cpu=1).__enter__()
        self.id = worker_id
        self.global_update_func = global_update_func
        self.env = env
        self.horizon = horizon
        self.sr_file_name = sr_file_name
        self.stochastic = stochastic
        self.sess = U.get_session()#U.make_session()
        if self.id == 0:
            self.policy_net = worker_pi_0
            self.discriminator_net = worker_disc_0
        elif self.id == 1:
            self.policy_net = worker_pi_1
            self.discriminator_net = worker_disc_1
        elif self.id == 2:
            self.policy_net = worker_pi_2
            self.discriminator_net = worker_disc_2
      
        #self.discriminator_net = reward_giver#TransitionClassifier(self.env, policy_net_hid_size, entcoeff=entcoeff, worker_id=self.id, scope="worker_adversary{}".format(self.id))

        self.initial_states = initial_states

        #if self.id == 0:

        print('worker :', self.id, 'generated', 'batch :', self.horizon)
        #print(self.id, tf.get_default_graph(), '\n\n\n\n')

        self.seg = None
        U.initialize()
        #self.update_local_model()

        threading.Thread.__init__(self)


    def run(self):
        global lock, pi
        with self.sess.as_default(), self.sess.graph.as_default():
            t = 0
            ac = self.env.action_space.sample()
            new = True
            rew = 0.
            true_rew = 0.0
            if True:#self.id == 0:
                ob = self.env.reset()
            else:
                self.env.reset()
                self.env.sim.set_state_from_flattened(random.choice(self.initial_states))
                self.env.sim.forward()
                ob = self.env.get_obs()

            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_true_ret = 0
            ep_true_rets = []
            ep_rets = []
            ep_lens = []

            obs = np.array([ob for _ in range(self.horizon)])
            true_rews = np.zeros(self.horizon, 'float32')
            rews = np.zeros(self.horizon, 'float32')
            vpreds = np.zeros(self.horizon, 'float32')
            news = np.zeros(self.horizon, 'int32')
            acs = np.array([ac for _ in range(self.horizon)])
            prevacs = acs.copy()

            while True:
                #print(self.id, 'ang data collectttttti', t)
                prevac = ac
                ac, vpred = self.policy_net.act(self.stochastic, ob)
                # Slight weirdness here because we need value function at time T
                # before returning segment [0, T-1] so we get the correct
                # terminal value
                if t > 0 and t % self.horizon == 0:
                    print(self.id, 'worker local update start')
                    self.seg = {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                        "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
                    """
                    yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                        "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
                    """
                    _, vpred = self.policy_net.act(self.stochastic, ob)
                    lock.acquire()
                    self.update_global_model()
                    #self.update_local_model()
                    lock.release()
                    # Be careful!!! if you change the downstream algorithm to aggregate
                    # several of these batches, then be sure to do a deepcopy
                    ep_rets = []
                    ep_true_rets = []
                    ep_lens = []
                i = t % self.horizon
                obs[i] = ob
                vpreds[i] = vpred
                news[i] = new
                acs[i] = ac
                prevacs[i] = prevac

                rew = self.discriminator_net.get_reward(ob, ac)
                ob, true_rew, new, _ = self.env.step(ac)

                rews[i] = rew + true_rew
                true_rews[i] = true_rew

                cur_ep_ret += rew
                cur_ep_true_ret += true_rew
                cur_ep_len += 1
                if new:
                    ep_rets.append(cur_ep_ret)
                    ep_true_rets.append(cur_ep_true_ret)
                    ep_lens.append(cur_ep_len)
                    cur_ep_ret = 0
                    cur_ep_true_ret = 0
                    cur_ep_len = 0
                    if True:#self.id == 0:
                        ob = self.env.reset()
                    else:
                        self.env.reset()
                        self.env.sim.set_state_from_flattened(random.choice(self.initial_states))
                        self.env.sim.forward()
                        ob = self.env.get_obs()
                t += 1


    def update_global_model(self):
        print('global update start')
        assert type(self.seg) != type(None)
        self.global_update_func(self.id, copy.deepcopy(self.seg))
        self.seg = None
        print('global update complete')


    def update_local_model(self):
        global pi, reward_giver
        U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(self.policy_net.get_variables(), pi.get_variables())])
        U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(self.discriminator_net.get_variables(), reward_giver.get_variables())])
        print('local update complete')


class Learner():
    def __init__(self, env, viewer, sr_file_name, num_workers, policy_net_hid_size, _reward_giver, expert_dataset, rank,
          pretrained, pretrained_weight, *,
          g_step, d_step, entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,
          callback=None):
        global reward_giver
        self.env = env
        self.viewer = viewer
        self.sr_file_name = sr_file_name
        self.policy_net_hid_size = policy_net_hid_size
        reward_giver = _reward_giver
        self.expert_dataset = expert_dataset
        self.rank = rank
        self.pretrained = pretrained
        self.pretrained_weight = pretrained_weight
        self.g_step = g_step
        self.d_step = d_step
        self.entcoeff = entcoeff
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.vf_stepsize = vf_stepsize
        self.d_stepsize = d_stepsize
        self.vf_iters = vf_iters
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.max_iters = max_iters
        self.callback = callback
        self.num_workers = num_workers
        self.timesteps_per_batch = timesteps_per_batch
        self.workers = []

        #print('global : ', tf.get_default_graph())


    def learn(self):
        global pi, reward_giver, worker_pi_0, worker_pi_1, worker_pi_2, worker_disc_0, worker_disc_1, worker_disc_2
        self.nworkers = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        np.set_printoptions(precision=3)
        # Setup losses and stuff
        # ----------------------------------------
        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        pi = mlp_policy.MlpPolicy(name="pi", ob_space=ob_space, ac_space=ac_space, reuse=(self.pretrained_weight != None), hid_size=self.policy_net_hid_size, num_hid_layers=2)
        oldpi = mlp_policy.MlpPolicy(name="oldpi", ob_space=ob_space, ac_space=ac_space, hid_size=self.policy_net_hid_size, num_hid_layers=2)
        worker_pi_0 = mlp_policy.MlpPolicy(name="worker_pi_0", ob_space=ob_space, ac_space=ac_space, hid_size=self.policy_net_hid_size, num_hid_layers=2)
        worker_pi_1 = mlp_policy.MlpPolicy(name="worker_pi_1", ob_space=ob_space, ac_space=ac_space, hid_size=self.policy_net_hid_size, num_hid_layers=2)
        worker_pi_2 = mlp_policy.MlpPolicy(name="worker_pi_2", ob_space=ob_space, ac_space=ac_space, hid_size=self.policy_net_hid_size, num_hid_layers=2)

        worker_disc_0 = TransitionClassifier(self.env, self.policy_net_hid_size, entcoeff=self.entcoeff, worker_id=-1, scope="worker_adversary0")
        worker_disc_1 = TransitionClassifier(self.env, self.policy_net_hid_size, entcoeff=self.entcoeff, worker_id=-1, scope="worker_adversary1")
        worker_disc_2 = TransitionClassifier(self.env, self.policy_net_hid_size, entcoeff=self.entcoeff, worker_id=-1, scope="worker_adversary2")

        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return


        ob = U.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])
   

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = self.entcoeff * meanent

        vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        surrgain = tf.reduce_mean(ratio * atarg)

        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

        dist = meankl

        all_var_list = pi.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
        vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
        assert len(var_list) == len(vf_var_list) + 1
        self.d_adam = MpiAdam(reward_giver.get_trainable_variables())
        self.vfadam = MpiAdam(vf_var_list)
        self.get_flat = U.GetFlat(var_list)
        self.set_from_flat = U.SetFromFlat(var_list)
        klgrads = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
        fvp = U.flatgrad(gvp, var_list)


        #print('pi :', pi.get_variables())
        self.assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        self.assign_workerpi0_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(worker_pi_0.get_variables(), pi.get_variables())])
        self.assign_workerpi1_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(worker_pi_1.get_variables(), pi.get_variables())])
        self.assign_workerpi2_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(worker_pi_2.get_variables(), pi.get_variables())])

        self.assign_workerdisc0_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(worker_disc_0.get_variables(), reward_giver.get_variables())])
        self.assign_workerdisc1_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(worker_disc_1.get_variables(), reward_giver.get_variables())])
        self.assign_workerdisc2_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(worker_disc_2.get_variables(), reward_giver.get_variables())])

        self.compute_losses = U.function([ob, ac, atarg], losses)
        self.compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
        self.compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
        self.compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))


        U.initialize()
        th_init = self.get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        self.set_from_flat(th_init)
        self.d_adam.sync()
        self.vfadam.sync()
        if self.rank == 0:
            print("Init param sum", th_init.sum(), flush=True)

        # Prepare for rollouts
        # ----------------------------------------
        #seg_gen = traj_segment_generator(pi, viewer, sr_file_name, expert_dataset, env, reward_giver, timesteps_per_batch, stochastic=True)
        #seg_gen_list = [agent.seg for agent in self.agents]

        self.episodes_so_far = 0
        self.timesteps_so_far = 0
        self.iters_so_far = 0
        tstart = time.time()
        self.lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
        self.true_rewbuffer = deque(maxlen=40)

        assert sum([self.max_iters > 0, self.max_timesteps > 0, self.max_episodes > 0]) == 1

        g_loss_stats = stats(self.loss_names)
        d_loss_stats = stats(reward_giver.loss_name)
        ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
        # if provide pretrained weight
        if self.pretrained_weight is not None:
            U.load_variables(self.pretrained_weight, variables=pi.get_variables())


        if self.num_workers > 1:
            intitial_states_list = self.get_workers_initial_states()
        else:
            intitial_states_list = [None]
        for i in range(self.num_workers):
            worker_env = JacoEnv()
            worker_env.seed(i * 10)
            #worker_generator = self.policy_func("pi{}".format(i), ob_space, ac_space, reuse=(self.pretrained_weight != None))
            worker = Worker(i, self.global_update_func, worker_env, self.policy_net_hid_size,
                  self.entcoeff, self.timesteps_per_batch, self.sr_file_name, intitial_states_list[i])
            self.workers.append(worker)

        for worker in self.workers:
            time.sleep(1)
            worker.start()

        """
        while True:
            if self.callback: callback(locals(), globals())
            if self.max_timesteps and self.timesteps_so_far >= self.max_timesteps:
                break
            elif self.max_episodes and self.episodes_so_far >= self.max_episodes:
                break
            elif self.max_iters and self.iters_so_far >= self.max_iters:
                break

            #logger.log("********** Iteration %i ************" % self.iters_so_far)
        """

    @contextmanager
    def timed(self, msg):
        if self.rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield
    def allmean(self, x):
            assert isinstance(x, np.ndarray)
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= self.nworkers
            return out
    def get_workers_initial_states(self):
        initial_stateidx_list = self.expert_dataset.initial_stateidx_list
        initial_state_list = [] #initial states in all traj (traj X phase)
        self.env.reset()
        demo_acs = self.expert_dataset.acs
        for ep in range(len(demo_acs)):
            initial_states = [] #initial states in one traj
            p = 0
            for i in range(len(demo_acs[ep])):
                if p == self.num_workers:
                    break
                if i == initial_stateidx_list[ep][p]:
                    initial_states.append(self.env.sim.get_state().flatten())
                    p += 1
                self.env.step(demo_acs[ep][i])
            initial_state_list.append(np.array(initial_states))
            self.env.reset()
        initial_state_list = np.array(initial_state_list)
        return [initial_state_list[:, i] for i in range(self.num_workers)]
        

    def global_update_func(self, worker_id, seg):
        global pi, reward_giver
        # ------------------ Update G ------------------
        def fisher_vector_product(p):
            return self.allmean(self.compute_fvp(p, *fvpargs)) + self.cg_damping * p
        logger.log("Optimizing Policy...")
        
        for _ in range(self.g_step):
            #with self.timed("sampling"):
            #    seg = seg_gen.__next__()
            self.add_vtarg_and_adv(seg, self.gamma, self.lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            self.vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            self.assign_old_eq_new()  # set old parameter values to new parameter values
            with self.timed("computegrad"):
                *lossbefore, g = self.compute_lossandgrad(*args)
            lossbefore = self.allmean(np.array(lossbefore))
            g = self.allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with self.timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters, verbose=self.rank == 0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / self.max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = self.get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    self.set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = self.allmean(np.array(self.compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > self.max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    self.set_from_flat(thbefore)
                if self.nworkers > 1 and self.iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with self.timed("vf"):
                for _ in range(self.vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                             include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy
                        g = self.allmean(self.compute_vflossandgrad(mbob, mbret))
                        self.vfadam.update(g, self.vf_stepsize)

        g_losses = meanlosses
        if worker_id == 0:
            self.assign_workerpi0_eq_new()
        if worker_id == 1:
            self.assign_workerpi1_eq_new()
        if worker_id == 2:
            self.assign_workerpi2_eq_new()
        #for (lossname, lossval) in zip(self.loss_names, meanlosses):
        #    logger.record_tabular(lossname, lossval)
        #logger.record_tabular("ev_tdlam_before", explained_variance(self.vpredbefore, tdlamret))
        
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        #logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = self.expert_dataset.get_worker_dset(worker_id).get_next_batch(len(ob))
        batch_size = len(ob) // self.d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                    include_final_partial_batch=False,
                                                    batch_size=batch_size):
            ob_expert, ac_expert = self.expert_dataset.get_worker_dset(worker_id).get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            self.d_adam.update(self.allmean(g), self.d_stepsize)
            d_losses.append(newlosses)
        #logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        
        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        self.true_rewbuffer.extend(true_rets)
        self.lenbuffer.extend(lens)
        self.rewbuffer.extend(rews)

        self.episodes_so_far += len(lens)
        self.timesteps_so_far += sum(lens)
        self.iters_so_far += 1

        #if self.rank == 0:
        #    logger.dump_tabular()
        if worker_id == 0:
            self.assign_workerdisc0_eq_new()
            evaluate(pi, self.viewer, self.sr_file_name, self.env, reward_giver, self.timesteps_per_batch, stochastic=True)
        if worker_id == 1:
            self.assign_workerdisc1_eq_new()
        if worker_id == 2:
            self.assign_workerdisc0_eq_new()

        assert seg != None


    def add_vtarg_and_adv(self, seg, gamma, lam):
        new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]
    


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
