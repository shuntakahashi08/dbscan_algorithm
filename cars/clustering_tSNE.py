import mglearn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import dbscan5
import time
from  sklearn.datasets import make_circles
import pickle
import numpy as np 


with open('new_tsne_data.pickle', 'rb') as f:
	tsne_data = pickle.load(f)
X=np.array(tsne_data)

#print(tsne_data)
#eps=0.7
#MinPts=2
eps=1.0
MinPts=3
dbscan = dbscan5.dbscan(X,eps,MinPts)
start = time.time()
threshold_expand=4
clusters = dbscan.clustering(threshold_expand)

#書き出し操作
with open('dbscan_obj.pickle', 'wb') as f:
	pickle.dump(dbscan, f)
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

print(clusters)

unique_cluster = set(clusters)
cluster_dict={}
for cluster in unique_cluster:
	cluster_dict[cluster]=[x for index,x in enumerate(X) if clusters[index]==cluster]

#print(cluster_dict)


plt.xlabel('val x', fontsize=20) # x軸ラベル
plt.ylabel('val y', fontsize=20) # y軸ラベル
#plt.grid(True) # 目盛線の表示

# グラフの描画
#k:cluster label y:data items():辞書に含まれるすべてのキーと組み合せを取得
for k, v in cluster_dict.items():
	x_array = np.array(v)
	plt.scatter(x_array[:,0],x_array[:,1],s=50,alpha=0.3,label=k)
	
plt.legend(bbox_to_anchor=(1.15,1),loc="upper right", fontsize=14) # (7)凡例表示
plt.show()
