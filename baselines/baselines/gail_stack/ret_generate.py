from jaco_arm import JacoPickUpEnv as JacoEnv
import mujoco_py
import gym
import numpy as np

env = JacoEnv()

traj_data = np.load('data/lift_demo.npz', allow_pickle=True)
obs = traj_data['obs']
acs = traj_data['acs']

ret_save_list = []
obs_save_list = []
acs_save_list = []

count = 0

for ep in range(len(obs)):
	true_rew_list = []
	_ = env.reset()	
	rew_sum = 0
	doned_step = -1
	for step in range(obs[ep].shape[0]):
		_, true_rew, done, ___ = env.step(acs[ep][step])
		rew_sum += true_rew
		true_rew_list.append(np.array([true_rew]))
		print(true_rew)
		if true_rew > 0:
			print(len(true_rew_list))
			doned_step = step
			count += 1
			break
	if doned_step == -1:
		continue

	if rew_sum >0:
		print('ep :', ep,'rew sum :', rew_sum)
		obs_save_list.append(obs[ep][:doned_step])
		acs_save_list.append(acs[ep][:doned_step])

	assert len(true_rew_list) == len(obs[ep][:doned_step])

	ret_save_list.append(np.array(true_rew_list))

	if count > 50:
		break

ret_save_list = np.array(ret_save_list)
obs_save_list = np.array(obs_save_list)
acs_save_list = np.array(acs_save_list)


np.savez('pickup_rew1.npz', obs = obs_save_list[:50], acs=acs_save_list[:50], rets=ret_save_list[:50])

