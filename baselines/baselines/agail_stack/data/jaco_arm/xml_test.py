from . import JacoStackEnv as JacoEnv
import mujoco_py
import gym
import numpy as np
import os

env = JacoEnv()


model_path = "stack0_1.xml"
fullpath = os.path.join(
            os.path.dirname(__file__), "assets", model_path)

print(fullpath)

mjpy_model = mujoco_py.load_model_from_path(fullpath)
sim = mujoco_py.MjSim(self.mjpy_model)
env.sim = sim

viewer = mujoco_py.MjViewer(env.sim)

while True:
	viewer.render()
