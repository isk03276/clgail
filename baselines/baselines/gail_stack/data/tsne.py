# Apply clustering instead of class names.
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

n = 16
data = np.load('stack_demo.npz')
obs = data['obs']
obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])[:10000]

tsne = TSNE(n_components=2,init='random',  random_state=0)

gmm = GMM(n_components=n)
model = gmm.fit(obs)
y = model.predict(obs)

reducted_obs = tsne.fit_transform(obs)
labeled_x = [[] for i in range(n)]

for i in range(len(obs)):
   labeled_x[y[i]].append(reducted_obs[i])
labeled_x = np.array(labeled_x)


###for capture
y_idx = 0
for i in range(len(y)):
	if y_idx == 10:
		break
	elif y_idx == y[i]:
		print(i, y_idx)
		y_idx += 1
##plot
marker_size = 13
for i in range(len(labeled_x)):
   temp_0 = []
   temp_1 = []
   for j in labeled_x[i]:
      temp_0.append(j[0])
      temp_1.append(j[1])
   sc = plt.scatter(temp_0, temp_1, marker_size) 
   #plt.colorbar(sc)
   """
   i_len = len(labeled_x[i])
   i_y = []
   for j in range(i_len):
      i_y.append([y[i]])
   plt.scatter(labeled_x[i], i_y)
   """
#plt.title('Expectation ')
plt.title('Stack demo data clustering')
plt.show()



