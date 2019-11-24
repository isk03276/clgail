from jaco_arm import JacoStackEnv as JacoEnv
import mujoco_py
import gym
import numpy as np

env = JacoEnv()
traj_limit = 30

traj_data = np.load('new_stack.npz', allow_pickle=True)
obs = traj_data['obs'][:traj_limit]
acs = traj_data['acs'][:traj_limit]

state_idx_list = []

for ep in range(traj_limit):
	sub_traj_len = int(len(obs[ep]) /3)
	state_idx_list.append(np.array([sub_traj_len-1, sub_traj_len*2-1]))
state_idx_list = np.array(state_idx_list)
np.savez('sub_traj_idx.npz', idx = state_idx_list)
print(state_idx_list.shape)

viewer = mujoco_py.MjViewer(env.sim)

test_state = None
for ep in range(traj_limit):
	_ = env.reset()	
	i = 0
	for step in range(obs[ep].shape[0]):
		if i == 2:
			break
		_, true_rew, __, ___ = env.step(acs[ep][step])
		viewer.render()
		if step == state_idx_list[ep][i]:
			test_state = env.sim.get_state().flatten()
			input('reset1?')
			break
	break

while True:
	env.reset()
	viewer.render()
	input('test2?')
	env.sim.set_state_from_flattened(test_state)
	print(env.sim.forward())
	viewer.render()
	input('reset3?')
		
