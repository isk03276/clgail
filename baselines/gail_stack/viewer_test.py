from jaco_arm import JacoStackEnv as JacoEnv
import mujoco_py

env=JacoEnv()
viewer = mujoco_py.MjViewer(env.sim)

while True:
	env.reset()
	viewer.render()
