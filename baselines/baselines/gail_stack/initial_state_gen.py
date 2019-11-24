import numpy as np
from jaco_arm import JacoPickUpEnv as JacoEnv
import mujoco_py

env = JacoEnv()
data = np.load('data/pickup_rew.npz', allow_pickle=True)
obs = data['obs'][:50]
acs = data['acs'][:50]

env.reset()
env.set_qpos_qvel

#viewer = mujoco_py.MjViewer(env.sim)

phase = 8



"""
#by len
for ep in range(len(obs)):
	#env.set_target_pos(obs[ep][0][-3:])
	env.reset()
	state_list = [env.sim.get_state().flatten()]
	#env.sim.model.geom_pos[1] = obs[18:21]
	sub_len = int(len(obs[ep]) / phase)
	count = 0
	for i in range(len(obs[ep])):
		ob, rew, __, ___ = env.step(acs[ep][i])
		if count == phase:
			break
		if i % sub_len == 0:
			state_list.append(env.sim.get_state().flatten())
			print(env.sim.get_state().flatten().shape)
			count += 1
		viewer.render()
	save_list.append(np.array(state_list))
np.savez('initial_states_bylen.npz', initial_states = np.array(save_list))
"""
def gen_bylen():
	save_list  = []
	for ep in range(len(obs)):
		#env.set_target_pos(obs[ep][0][-3:])
		env.reset()
		state_list = []#[env.sim.get_state().flatten()]
		#env.sim.model.geom_pos[1] = obs[18:21]
		sub_len = int(len(obs[ep]) / phase)
		count = 0
		for i in range(len(obs[ep])):
			ob, rew, __, ___ = env.step(acs[ep][i])
			if count == phase:
				break
			if i % sub_len == 0:
				state_list.append(env.sim.get_state().flatten())
				print(env.sim.get_state().flatten().shape)
				count += 1
			#viewer.render()
		save_list.append(np.array(state_list))
	np.savez('pickup_initial_states_bylen.npz', initial_states = np.array(save_list))

def gen_bydist():
	save_list = []
	for ep in range(len(obs)):
		env.reset()
		state_list = []#[env.sim.get_state().flatten()]
		state_dist_dict = dict()
		state_candidate_list = []
		for t in range(len(obs[ep])):
			goal_state = obs[ep][-1]
			state_candidate_list.append(env.sim.get_state().flatten())
			_, rew, __, ___ = env.step(acs[ep][t])
			state_dist_dict[t] = np.linalg.norm(goal_state - obs[ep][t])
		sorted_tuple = sorted(state_dist_dict.items(), key=lambda i : i[1])
		print(sorted_tuple)
		initial_state_idx = int(len(sorted_tuple)/(phase+1))
		for i in range(phase):
			state_list.append(state_candidate_list[sorted_tuple[initial_state_idx*(i+1)][0]])
		save_list.append(np.array(state_list))
	save_list = np.array(save_list)
	np.savez('pickup_initial_states_bydist.npz', initial_states = np.array(save_list))

gen_bylen()