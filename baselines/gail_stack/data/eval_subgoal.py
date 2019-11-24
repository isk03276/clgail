import numpy as np
import mujoco_py
import gym
from jaco_arm import JacoStackEnv as JacoEnv

env = JacoEnv()

viewer = mujoco_py.MjViewer(env.sim)
expert_data = np.load('stack.npz', allow_pickle=True)
obs = expert_data['obs'][5]
acs = expert_data['acs'][5]

sub_goal_idx = [6, 15, 24]

env.reset()

for i in range(28):
	env.step(acs[i])
	viewer.render()
	
	if i in sub_goal_idx:
		input('continue?')
	




