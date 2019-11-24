"""
sub goal에 해당되는 demo들을 얻기위해 각 expert data trajectory에서 가장 유사한 state를 구하고 해당 state이전의 trajectory 반환
error result : +1
"""
import numpy as np
import mujoco_py
import gym
from jaco_arm import JacoStackEnv as JacoEnv
import math

env = JacoEnv()

viewer = mujoco_py.MjViewer(env.sim)
expert_data = np.load('stack.npz', allow_pickle=True)

obs = expert_data['obs'][25]
baseline = obs[20]

for i in range(len(obs)):
	dist_sum = 0
	for j in range(len(obs[i])):
		dist_sum += math.sqrt((obs[i][j] - baseline[j])**2)





