from jaco_arm import JacoStackEnv as JacoEnv
import mujoco_py
import gym
import numpy as np
import glfw
from sklearn.mixture import GaussianMixture as GM
import cv2
from sklearn.decomposition import PCA

env = JacoEnv()

traj_data = np.load('new_stack.npz', allow_pickle=True)
obs = traj_data['obs'][:30]
acs = traj_data['acs'][:30]

ret_save_list = []

#pca = PCA(n_components=3)
#nobs = pca.fit_transform(np.vstack(obs))

print(np.vstack(obs).shape)
print(obs[0].shape)
gm = GM(n_components=3, init_params='random', random_state=0)
gm.fit(np.vstack(obs))
for i in range(len(obs)):
	print('traj [', i, '] :', gm.predict(obs[i]))


#np.savez('new_stack.npz', obs = obs[:100], acs=acs[:100], rets=ret_save_list)

