import gym
import mujoco_py
from jaco_arm2 import Jaco2PickUpEnv as JacoEnv
import numpy as np

env = JacoEnv()

env.reset()
viewer = mujoco_py.MjViewer(env.sim)
env.sim.data.qpos[0] = 0#4.8046
env.sim.data.qpos[1] = 2.9#2.9248
env.sim.data.qpos[2] = 1.2987#1.002
env.sim.data.qpos[3] = -2.079#4.203
env.sim.data.qpos[4] = 1.4017#1.4458
env.sim.data.qpos[5] = 0.0#1.3233
env.sim.data.qpos[6] = 1.0#0.0
env.sim.data.qpos[7] = 1.0#0.0
env.sim.data.qpos[8] = 1.0#0.0
env.sim.forward()

while True:
	viewer.render()
