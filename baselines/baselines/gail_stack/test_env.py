import gym
import mujoco_py
from jaco_arm import JacoStackEnv as JacoEnv
import numpy as np
import random

env = JacoEnv()

env.reset()
viewer = mujoco_py.MjViewer(env.sim)

demo_data = np.load('data/stack_rew.npz', allow_pickle=True)
#initial_states_dist = np.load('data/jaco2_initial_states8.npz', allow_pickle=True)['initial_states']
#initial_states_list = initial_states_dist[:, 0]

acs = demo_data['acs']
obs = demo_data['obs']

ep = 0
while True:
	#initial_state = random.choice(initial_states_list)
	env.reset()
	#env.sim.set_state_from_flattened(initial_state)
	#env.sim.forward()
	rew_sum = 0
	for i in range(len(acs[ep])):
		viewer.render()
		new_ob, rew, done, info = env.step(acs[ep][i])
		if rew>0:
			print('ep :', ep, 'rew :', rew)
	ep += 1

