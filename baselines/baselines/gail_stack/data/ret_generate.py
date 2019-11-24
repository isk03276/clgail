from jaco_arm import JacoEnv as JacoEnv
import mujoco_py
import gym
import numpy as np

env = JacoEnv()

traj_data = np.load('data/lift_demo.npz', allow_pickle=True)
obs = traj_data['obs']
acs = traj_data['acs']

ret_save_list = []
for ep in range(50):
	true_rew_list = []

	_ = env.reset()	
	for step in range(obs[ep].shape[0]):
		_, true_rew, __, ___ = env.step(acs[ep][step])

		true_rew_list.append(np.array([true_rew]))
	true_rew_list = np.array(true_rew_list)
	ret_save_list.append(true_rew_list)
ret_save_list = np.array(ret_save_list)

np.savez('new_pickup.npz', obs = obs[:50], acs=acs[:50], rets=ret_save_list)

