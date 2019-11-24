from jaco_arm import JacoStackEnv as JacoEnv
import mujoco_py
import gym
import numpy as np
import os




model_path = "stack2_1.xml"
fullpath = os.path.join(
            os.path.dirname(__file__), "jaco_arm/assets", model_path)

mjpy_model = mujoco_py.load_model_from_path(fullpath)
sim = mujoco_py.MjSim(mjpy_model)
env = JacoEnv(sim)
env.sim.reset()
#env.sim = sim
#sim.forward()

viewer = mujoco_py.MjViewer(env.sim)

while True:
	viewer.render()
