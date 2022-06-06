from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import dbscan5
import time
from  sklearn.datasets import make_circles
import pickle
import numpy as np


with open('moons_dbscan_obj.pickle', 'rb') as f:
	dbscan = pickle.load(f)

pre_clusters = dbscan.clusterNo
clusters = [i for i in pre_clusters if i != -1]
clusters = [i for i in clusters if i != 9999]
#set:重複しない要素
original_no_of_clusters=len(set(clusters))
#print(clusters)
no_of_cluster=2
#delta_minpts=1
new_minpts=3
#new_eps=0.15
new_eps=0.15

"""
for cls in reversed(range(original_no_of_clusters)):
#cls=2
	clusters,X = dbscan.squeezing(cls,new_minpts,new_eps)
	##重複しない要素	
	unique_cluster = set(clusters)
	cluster_dict={}
	for cluster in unique_cluster:
		cluster_dict[cluster]=[x for index,x in enumerate(X) if clusters[index]==cluster]
		
	##collections.Counter():降順かつその値の数を返す {値:個数,値:個数,...} {key:value,...}
	for k, v in cluster_dict.items():
		x_array = np.array(v)
		plt.scatter(x_array[:,0],x_array[:,1],s=50,alpha=0.3,label=k)
	plt.title('No. of cluster :{0}'.format(cls))
	plt.legend(bbox_to_anchor=(1.15,1),loc="upper right", fontsize=14) # (7)凡例表示
	plt.show()
	if cls == no_of_cluster:
		break
"""


#有効クラスタのみをプロット
import collections
class_count = collections.Counter(pre_clusters)
#if 9999 in class_count:
#	del class_count[9999]
#if -1 in class_count:
#	del class_count[-1]

class_target=[]
for i,(k, v) in enumerate(sorted(class_count.items(), key=lambda x: -x[1])):
	if 2 > i and k != 9999 and k != -1:
		class_target.append(k)
#print(class_target)
clusters = [i if i in class_target else -1 for i in pre_clusters]
#print(len(clusters))

from sklearn.datasets import make_moons
X,y= make_moons(n_samples=500, noise=0.15, random_state=3)

unique_cluster = set(clusters)
#print(unique_cluster)
cluster_dict={}
for cluster in unique_cluster:
	cluster_dict[cluster]=[x for index,x in enumerate(X) if clusters[index]==cluster]
#print(clusters)
#print(cluster_dict)


for k, v in cluster_dict.items():
	x_array = np.array(v)

	plt.scatter(x_array[:,0],x_array[:,1],s=50,alpha=0.3,label=k)
#plt.title('No. of cluster :2')
plt.legend(bbox_to_anchor=(1.15,1),loc="upper right", fontsize=14) # (7)凡例表示
plt.show()
