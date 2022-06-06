import shutil
import glob
import numpy as np
from skimage import io, color,transform
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import random
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm
import cv2
import time

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

np.random.seed(0)
#---------------------------------画像をグレースケールベクトル行列に変換-------------------------------
def get_gray_scale(label_list,dataDir_list):
	
	labels=[]
	data_grayscales=[]
	img_list=[]
	for label,dataDir in zip(label_list,dataDir_list):
		#ワイルドカードでdataディレクトリ内の全ファイル名を取得してリスト化
		files = glob.glob(dataDir + "*")
		for file in files:	
			labels.append(label)
			#バイナリーデータとして読み込む
			img = io.imread(file)
			img_list.append(img)
			#元がgray_scaleであるため
			#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#			大きさ統一
			#gray_resize = cv2.resize(gray,(32,32))
			gray_resize = cv2.resize(img,(32,32))
			data_grayscales.append(gray_resize)
	
	#各画像のHOG特徴量行列を要素とするリスト
	labels, data_grayscales = np.array(labels), np.array(data_grayscales)
	return labels, data_grayscales,img_list


car0Dir = './car/img/0/'
car1Dir = './car/img/1/'
car2Dir = './car/img/2/'
dataDir_list=[car1Dir,car2Dir]
label_list=[1,2]

carsum1Dir = './car/sum/1/'
carsum2Dir = './car/sum/2/'
sum_dataDir_list=[carsum1Dir,carsum2Dir]
sum_label_list=[1,2]

labels, data_grayscales, img_list = get_gray_scale(sum_label_list,sum_dataDir_list)

imglines_set =[]
for img in data_grayscales:
	#print(img.shape) (400,300)
	imglines =[]
	for imgline in img:
		imglines.extend(imgline)

	imglines_set.append(imglines)

imglines_set_np = np.array(imglines_set)	
	
img_sets = np.vstack(imglines_set_np)

#-----------t-sne
model = TSNE(n_components=2,perplexity=50,learning_rate=50)
tsne_result = model.fit_transform(img_sets)

print(tsne_result)

"""
#---------data_processing
tsne_result = np.array([j for label,j in zip(labels,tsne_result) if label ==0 or label ==1 or label==2])
print(tsne_result)

unique_cluster = set(labels)
cluster_dict={}
for cluster in unique_cluster:
	cluster_dict[cluster]=[x for index,x in enumerate(tsne_result) if labels[index]==cluster]
	


#plt.xlabel('val x', fontsize=20) # x軸ラベル
#plt.ylabel('val y', fontsize=20) # y軸ラベル
#plt.grid(True) # 目盛線の表示

# グラフの描画
#k:cluster label y:data items():辞書に含まれるすべてのキーと組み合せを取得
for k, v in cluster_dict.items():
	x_array = np.array(v)
	plt.scatter(x_array[:,0],x_array[:,1],s=50,alpha=0.3,label=k)

plt.title('No. of cluster :2')	
plt.legend(bbox_to_anchor=(1.15,1),loc="upper right", fontsize=14) # (7)凡例表示
plt.show()
"""

#pickle 保存
#import pickle

#with open('new_tsne_data.pickle', 'wb') as f:
#    pickle.dump(tsne_result, f)
	
#with open('dbscan_obj.pickle', 'rb') as f:
#	hoge = pickle.load(f)
#print(hoge)