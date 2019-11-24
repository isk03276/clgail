import gym
import mujoco_py
from jaco_arm import JacoPickUpEnv as JacoEnv
import numpy as np

env = JacoEnv()

env.reset()
viewer = mujoco_py.MjViewer(env.sim)

demo_data = np.load('data/pickup_rew.npz', allow_pickle=True)
acs = demo_data['acs']
obs = demo_data['obs']

ep = 0
print(acs[1][1].shape)
while True:
	#env.set_target_pos(obs[ep][ep][-3:])
	ob = env.reset()
	rew_sum = 0
	for i in range(len(acs[ep])):
		viewer.render()
		new_ob, rew, done, info = env.step(acs[ep][i])
		rew_sum += rew
		if done:
			break
	input('rew_sum :' + str(rew_sum))
	ep += 1

