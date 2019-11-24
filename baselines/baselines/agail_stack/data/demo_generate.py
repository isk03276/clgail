from jaco_arm2 import Jaco2PickUpEnv as JacoEnv
import mujoco_py
import gym
import numpy as np

env = JacoEnv()

viewer = mujoco_py.MjViewer(env.sim)
demo = np.load('action.npz', allow_pickle=True, encoding='latin1')
acs = demo['acs']
print(acs[0].shape)

env.reset()
for ac in acs:
	env.step(ac[0])
	viewer.render()

