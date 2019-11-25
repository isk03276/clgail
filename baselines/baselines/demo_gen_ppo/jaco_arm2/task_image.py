import mujoco_py
import numpy as py
from jaco_arm2 import JacoPickUp as JacoEnv

env = JacoEnv()
viewer = mujoco_py.MjViewer(env.sim)

while True:
	env.reset()
	veiwer.render()
