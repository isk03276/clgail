'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np
import math
from baselines.gail_stack.curriculum import SubGoalGenerator


class Dset(object):
    def __init__(self, curriculum_obs, curriculum_acs, knowledge_obs, knowledge_acs, randomize, knowledge_ratio = 0.1):
        self.curriculum_obs = curriculum_obs
        self.curriculum_acs = curriculum_acs
        self.knowledge_obs = knowledge_obs
        self.knowledge_acs = knowledge_acs
        assert len(self.curriculum_obs) == len(self.curriculum_acs)
        self.randomize = randomize
        self.use_knowledge = True if type(self.knowledge_obs) != type(None) else False
        self.num_curriculum_pairs = len(curriculum_obs)
        self.num_knowledge_pairs = len(knowledge_obs) if self.use_knowledge else 0
        self.knowledge_ratio = knowledge_ratio
        self.init_curriculum_pointer()
        if self.use_knowledge:
            self.init_knowledge_pointer()

    def init_curriculum_pointer(self):
        self.curriculum_pointer = 0
        if self.randomize:
            idx = np.arange(self.num_curriculum_pairs)
            np.random.shuffle(idx)
            self.curriculum_obs = self.curriculum_obs[idx, :]
            self.curriculum_acs = self.curriculum_acs[idx, :]

    def init_knowledge_pointer(self):
        self.knowledge_pointer = 0
        if self.randomize:
            idx = np.arange(self.num_knowledge_pairs)
            np.random.shuffle(idx)
            self.knowledge_obs = self.knowledge_obs[idx, :]
            self.knowledge_acs = self.knowledge_acs[idx, :]

    def get_next_batch(self, batch_size):
        if self.use_knowledge:
            if batch_size <0:
                return np.vstack((self.curriculum_obs, knowledge_obs)), np.vstack((self.curriculum_acs, self.knowledge_acs))
            knowledge_batch_size = int(batch_size * self.knowledge_ratio)
            curriculum_batch_size = batch_size - knowledge_batch_size
            if self.curriculum_pointer + curriculum_batch_size >= self.num_curriculum_pairs:
                self.init_curriculum_pointer()
            if self.knowledge_pointer + knowledge_batch_size >= self.num_knowledge_pairs:
                self.init_knowledge_pointer()            
            curriculum_end = self.curriculum_pointer + curriculum_batch_size
            knowledge_end = self.knowledge_pointer + knowledge_batch_size
            curriculum_obs = self.curriculum_obs[self.curriculum_pointer:curriculum_end, :]
            curriculum_acs = self.curriculum_acs[self.curriculum_pointer:curriculum_end, :]
            knowledge_obs = self.knowledge_obs[self.knowledge_pointer:knowledge_end, :]
            knowledge_acs = self.knowledge_acs[self.knowledge_pointer:knowledge_end, :]
            self.curricurm_pointer = curriculum_end
            self.knowledge_pointer = knowledge_end
            return np.vstack((curriculum_obs, knowledge_obs)), np.vstack((curriculum_acs, knowledge_acs))
        else:
            if batch_size < 0:
                return self.curriculum_obs, self.curriculum_acs
            if self.curriculum_pointer + batch_size >= self.num_curriculum_pairs:
                self.init_curriculum_pointer()
            end = self.curriculum_pointer + batch_size
            curriculum_obs = self.curriculum_obs[self.curriculum_pointer:end, :]
            curriculum_acs = self.curriculum_acs[self.curriculum_pointer:end, :]
            self.curriculum_pointer = end
            return curriculum_obs, curriculum_acs


class Mujoco_Dset(object):
    def __init__(self, expert_path, num_of_goal, train_fraction=0.7, traj_limitation=30, randomize=True):
        traj_data = np.load(expert_path, allow_pickle=True)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        self.total_obs = traj_data['obs'][:traj_limitation]
        self.total_acs = traj_data['acs'][:traj_limitation]
        self.traj_num = len(self.total_obs)

        self.randomize = randomize
        
        self.num_of_goal = num_of_goal
        self.init_curriculum()
        self.update_goal()

        #self.num_transition = len(self.obs)
        
        #self.dset = None#Dset(self.total_obs, self.total_acs, self.randomize)
        """
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)
        """
    def init_curriculum(self):
        self.subGoalGenerator = SubGoalGenerator(self.total_obs[0][0].shape[0])
        print('%%%%%%%%%%%%%%%!@#!@!!!!!!!!!!!!!!!!!!!!!!!%?')
        self.subGoalGenerator.train(self.total_obs, learning_iteration = 100)
        #self.current_sub_goals = None
        self.sub_goal_list = self.subGoalGenerator.create_sub_goals(self.total_obs, self.num_of_goal) #row : ep, col : sub goal
        print(self.sub_goal_list)

        self.current_phase = -1 #initial current goal phase
        self.curriculum_obs = [] #row : phase, col : sub trajectory
        self.curriculum_acs = [] #row : phase, col : sub trajectory

        self.prev_curriculum_obs = []
        self.prev_curriculum_acs = []

        #curriculum demo data
        for phase in range(self.num_of_goal):
            sub_traj_obs_list = []
            sub_traj_acs_list = []
            for ep in range(self.traj_num):
                start_idx = 0 if phase ==0 else self.sub_goal_list[ep][phase - 1]
                end_idx = self.sub_goal_list[ep][phase]
                sub_traj_obs_list.append(self.total_obs[ep][start_idx:end_idx])
                sub_traj_acs_list.append(self.total_acs[ep][start_idx:end_idx])
            self.curriculum_obs.append(np.vstack(np.array(sub_traj_obs_list)))
            self.curriculum_acs.append(np.vstack(np.array(sub_traj_acs_list)))
        sub_traj_obs_list = []
        sub_traj_acs_list = []
        for ep in range(self.traj_num): #append sub traj for original goal
            sub_traj_obs_list.append(self.total_obs[ep][self.sub_goal_list[ep][-1]:])
            sub_traj_acs_list.append(self.total_acs[ep][self.sub_goal_list[ep][-1]:])
        self.curriculum_obs.append(np.vstack(np.array(sub_traj_obs_list)))
        self.curriculum_acs.append(np.vstack(np.array(sub_traj_acs_list)))
        self.curriculum_obs = np.array(self.curriculum_obs)
        self.curriculum_acs = np.array(self.curriculum_acs)

        assert self.num_of_goal == self.curriculum_obs.shape[0] - 1 == self.curriculum_acs.shape[0] - 1

        demo_initial_state = []
        for states in self.total_obs:
            demo_initial_state.append(states[0])
        self.demo_initial_states = np.array(demo_initial_state)

        
    def data_flatten(self, obs, acs):
        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        if len(acs) > 2:
            acs = np.squeeze(acs)
        assert len(obs) == len(acs)

        return np.vstack(obs), np.vstack(acs)

    def update_goal(self):
        print('Sub goal updating..')
        self.current_phase += 1
        value_net_learning_obs = []
        if self.current_phase == self.num_of_goal - 1:
            for ep in range(self.traj_num):
                value_net_learning_obs.append(self.total_obs[ep][:])
        elif self.current_phase < self.num_of_goal:
            for ep in range(self.traj_num):
                value_net_learning_obs.append(self.total_obs[ep][:self.sub_goal_list[ep][self.current_phase]])
        else:
            assert True
        print('Value net training ( current phase :', self.current_phase, ')')

        self.subGoalGenerator.train(np.array(value_net_learning_obs), learning_iteration = 100)
        if self.current_phase == 0:
            self.knowledge_obs = None
            self.knowledge_acs = None
        elif self.current_phase == 1:
            self.knowledge_obs = np.vstack(self.curriculum_obs[self.current_phase - 1])
            self.knowledge_acs = np.vstack(self.curriculum_acs[self.current_phase - 1])
        else:
            self.knowledge_obs = np.vstack(self.curriculum_obs[:self.current_phase - 1])
            self.knowledge_acs = np.vstack(self.curriculum_acs[:self.current_phase - 1])
        self.dset = Dset(self.curriculum_obs[self.current_phase], self.curriculum_acs[self.current_phase], self.knowledge_obs, self.knowledge_acs, self.randomize)

        demo_score_sum = 0
        for ep in range(self.traj_num):
            if self.current_phase == self.num_of_goal -1:
                demo_score_sum += self.subGoalGenerator.predict_state_score(self.total_obs[ep][-1])
            else:
                demo_score_sum += self.subGoalGenerator.predict_state_score(self.total_obs[ep][self.sub_goal_list[ep][self.current_phase]])
        self.demo_score = demo_score_sum / self.traj_num #mean
        #self.dset.get_next_batch(100) #test@@@@@@@@@@@@@@@@

    def get_demo_score(self):
        return self.demo_score

    def get_demo_initial_states(self):
        return self.demo_initial_states

    def get_learner_score(self, states):
        learner_score_sum = 0
        num_state = len(states)
        assert num_state == self.traj_num
        for state in range(len(states)):
            learner_score_sum += self.subGoalGenerator.predict_state_score(state)
        return learner_score_sum / num_state

    def get_lenlist_goaltraj(self):
        return np.array(self.sub_goal_list)[:, self.current_phase]

    def get_goal_reward(self, state):
        return 0.1 / (1 +np.exp(-self.subGoalGenerator.predict_state_score(state)))

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError


def test(expert_path, traj_limitation):
    dset = Mujoco_Dset(expert_path, 3)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/stack.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation)
