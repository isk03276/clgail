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
img_save_list = []

viewer = mujoco_py.MjViewer(env.sim)
#viewer = mujoco_py.MjRenderContextOffscreen(env.sim, 0)
for ep in range(30):
	true_rew_list = []
	img_list = []

	_ = env.reset()	
	for step in range(obs[ep].shape[0]):
		#img_list.append(cv2.cvtColor(viewer._read_pixels_as_in_window(600, 600), cv2.COLOR_BGR2GRAY))
		img_list.append(viewer._read_pixels_as_in_window(800, 800).reshape(800*800*3))
		_, true_rew, __, ___ = env.step(acs[ep][step])
		#viewer._read_pixels_as_in_window(800, 600)
		#viewer.render()
		#rint(mjviewer._read_pixels_as_in_window())
		#print(env.sim.render())

		#true_rew_list.append(np.array([true_rew]))
	#true_rew_list = np.array(true_rew_list)
	img_list = np.array(img_list)
	img_save_list.append(img_list)
	#ret_save_list.append(true_rew_list)
img_save_list = np.array(img_save_list)
#ret_save_list = np.array(ret_save_list)

#np.savez('img.npz', obs=img_save_list, acs = acs)
obs = img_save_list
#print(obs[0].shape)
#cv2.imshow('plz', obs[0][0])

#cv2.waitKey(0)
#cv2.destroyAllWindows()

pca = PCA(n_components=3)
pca.fit(np.vstack(obs))


gm = GM(n_components=3, init_params='random', random_state=0)
gm.fit(pca.transform(np.vstack(obs)))
for i in range(len(obs)):
	nobs = pca.transform(obs[i])
	print('traj [', i, '] :', gm.predict(pca.transform(obs[i])))


#np.savez('new_stack.npz', obs = obs[:100], acs=acs[:100], rets=ret_save_list)

