'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, rets):
        self.inputs = inputs
        self.labels = labels
        self.rets = rets
        assert len(self.inputs) == len(self.labels) == len(self.rets)
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels, self.rets
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        rets = self.rets[self.pointer:end, :]
        self.pointer = end
        return inputs, labels, rets


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1):
        traj_data = np.load(expert_path, allow_pickle=True)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]
        #rets = traj_data['rets'][:traj_limitation]
 
        rets = []
        for ep in range(len(obs)):
            #temp = np.zeros(obs[ep].shape[0])
            temp = []
            temp.append(np.array([1.0]))
            for t in range(1, len(obs[ep])):
                temp.append(np.array([temp[t-1][0] * 0.99]))
            rets.append(np.array(list(reversed(temp))))
        rets = np.array(rets)
                
        self._obs = obs
        self._acs = acs
        self._rets = rets

        self.num_phases = 8

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        if False:#len(obs.shape) > 2:
            self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
            self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])
        else:
            self.obs = np.vstack(obs)
            self.acs = np.vstack(acs)
            self.rets = np.vstack(rets)

        #self.rets = traj_data['ep_rets'][:traj_limitation]
        #self.avg_ret = sum(self.rets)/len(self.rets)
        #self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.dset = Dset(self.obs, self.acs, self.rets)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.rets[:int(self.num_transition*train_fraction), :])
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.rets[int(self.num_transition*train_fraction):, :])

        self.make_worker_dset_bylen()
        self.log_info()

    def cal_phase(self):
        initial_stateidx_list = [[0] for _ in ragne(len(self._obs))]
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

    def make_worker_dset_bylen(self):
        if self.num_phases == 1:
            self.initial_state_list = initial_stateidx_list = [[0] for _ in ragne(len(self._obs))]
        else:
            self.initial_stateidx_list = []
            for ep in range(len(self._obs)):
                sub_traj_len = int(len(self._obs[ep]) /self.num_phases)
                ep_initial_state_list = [sub_traj_len*phase for phase in range(self.num_phases)]
                ep_initial_state_list.append(len(self._obs[ep]))
                ep_initial_state_list = np.array(ep_initial_state_list)
                self.initial_stateidx_list.append(ep_initial_state_list)
            self.initial_stateidx_list = np.array(self.initial_stateidx_list)

            
            self.dset_list = []
            for i in range(self.num_phases):
                obs = []
                acs = []
                rets = []
                for ep in range(len(self._obs)):
                    obs.append(self._obs[ep][self.initial_stateidx_list[ep][i]:])
                    acs.append(self._acs[ep][self.initial_stateidx_list[ep][i]:])
                    rets.append(self._rets[ep][self.initial_stateidx_list[ep][i]:])
                self.dset_list.append(Dset(np.vstack(np.array(obs)), np.vstack(np.array(acs)), np.vstack(np.array(rets))))
            

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        #logger.log("Average returns: %f" % self.avg_ret)
        #logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch_cur(self, batch_size, phase):
        if phase == 8 or phase == 0:
            print('##################size :', len(self.dset_list), 'phase :', phase)
            return self.dset_list[phase].get_next_batch(batch_size)
        else:
            current_batch_size = int((batch_size)*0.7)
            next_batch_size = batch_size - current_batch_size
            current_batch = self.dset_list[phase].get_next_batch(current_batch_size)
            next_batch = self.dset_list[phase-1].get_next_batch(next_batch_size)
            ob_batch = np.concatenate((current_batch[0], next_batch[0]), axis=0)
            ac_batch = np.concatenate((current_batch[1], next_batch[1]), axis=0)
            ret_batch = np.concatenate((current_batch[2], next_batch[2]), axis=0)
        
            return ob_batch, ac_batch, ret_batch        

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'curriculum':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        #plt.hist(self.rets)
        #plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
