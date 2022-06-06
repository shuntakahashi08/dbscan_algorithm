import numpy as np
from scipy import stats
import math
import collections


class dbscan:

	def __init__(self,D,eps,MinPts):
		
		self.eps = eps
		self.minpts = MinPts
		self.dataset =D
		#全データのクラスタ番号を初期化する
		self.clusterNo = [-1]*len(self.dataset)
		
	def clustering(self,threshold_expand):
	
		C=0
		
	
		def regionQuery(index):
			
			#距離計算する際，自分自身とも比較する
			#データ自体でなくて，datasetのindexを取得する
			neighbors = [ i for i,neighbor in enumerate(self.dataset) if np.linalg.norm(self.dataset[index]-neighbor)<=self.eps]
			return neighbors	
		
		def expandCluster(neighborPts,C,threshold_expand):
			prev_neighborPts = neighborPts
			ex_neighborPts=[]
			
			#neighborPtsのindexが入っている
			for np_index in neighborPts:
			
				#print(np_index)
				if self.clusterNo[np_index] <0:
				
					self.clusterNo[np_index] = C					
					expand_neighborPts = regionQuery(np_index)
					#上記の円は領域拡大を意味する．minpts以上の点があれば領域を拡大を認める
					if len(expand_neighborPts) >= self.minpts:
						
						#visitしたポイントを除外して配列を作成する
						expand_neighborPts = [n_point for n_point in expand_neighborPts ]#if self.clusterNo[n_point] == -1]						
						ex_neighborPts+=expand_neighborPts
						
						
			if len(ex_neighborPts) == 0:
				
				#再帰処理では必ず，終了条件を満たしたらreturnするように記述する．ただし，本来的に返したい処理結果は，self.clusterNoなので，ここでは何をreturnしてもよい
				return 1
						
			self.expand_counter+=1
			
			return expandCluster(ex_neighborPts,C,threshold_expand)
			
			
		#datasetは，行方向がサンプル，列方向が特徴量次元　任意の次元数でよい
		for index,p in enumerate(self.dataset):
			if self.clusterNo[index]<0:
				
				#data自体でなくて，dataのindexを渡すとdataのindexが戻る
				neighborPts = regionQuery(index)
				if len(neighborPts) < self.minpts:
					self.clusterNo[index] = 9999 #p is noise visitしたことになるので，クラスタ番号は9999で永続化する
				else:
					C+=1
					self.clusterNo[index] = C
					
					#1つのクラスタ番号の領域は以下の処理で完結する(expandClusterを呼び出す都度クラスタ番号を更新する）
					self.expand_counter=0
					expandCluster(neighborPts,C,threshold_expand)
					
					
					if self.expand_counter < threshold_expand:
						print(self.expand_counter)
						
						self.clusterNo= [-1 if i ==C else i for i in self.clusterNo]
						
		
		return self.clusterNo



	

	def squeezing(self,no_of_cluster,new_minpts,new_eps):
		self.eps = new_eps
		self.minpts=new_minpts
		def regionQuery(index):
			
			#距離計算する際，自分自身とも比較する
			#データ自体でなくて，datasetのindexを取得する
			neighbors = [ i for i,neighbor in enumerate(self.dataset) if np.linalg.norm(self.dataset[index]-neighbor)<=self.eps]
			return neighbors	
		
		
		def expandCluster(neighborPts,C):
			prev_neighborPts = neighborPts
			ex_neighborPts=[]
			
			#neighborPtsのindexが入っている
			for np_index in neighborPts:
			
				#print(np_index)
				if self.clusterNo[np_index] <0:
				#if self.clusterNo[np_index] ==-1 or self.clusterNo[np_index] ==-2:
					print('prev class',self.clusterNo[np_index])
					self.clusterNo[np_index] = C
					print('update class',self.clusterNo[np_index])
					#eps半径内にある任意の点np_indexを中心として，eps半径を描き，円内の点を探す その点のindexが返ってくる この時は，visitした点も改めてカウントする
					#visitした点が別のクラスタに属することもある					
					expand_neighborPts = regionQuery(np_index)
					#上記の円は領域拡大を意味する．minpts以上の点があれば領域を拡大を認める
					if len(expand_neighborPts) >= self.minpts:
						
						#visitしたポイントを除外して配列を作成する
						expand_neighborPts = [n_point for n_point in expand_neighborPts if self.clusterNo[n_point] == -1]
						
						ex_neighborPts+=expand_neighborPts
						
			
			if len(ex_neighborPts) == 0:
				
				return 1
				#ex_neighborPtsの平均的な数を計算し、それを大きく下回った閾値になったら脱出
				
			return expandCluster(ex_neighborPts,C)
		
		#要素数の大きいクラスを上位nまで特定する
		def chunking():
			
			print('minpts:',self.minpts)
			for index,p in enumerate(self.dataset):
				#print(index,self.eps,self.minpts)
				#有望クラスタラベルのデータだったとき
				if self.clusterNo[index] in class_target:
					#print('clusterno:',self.clusterNo[index],class_target)
					neighborPts = regionQuery(index)
					if len(neighborPts) >= self.minpts:
						C=self.clusterNo[index]
						
						expandCluster(neighborPts,C)
			class_count = collections.Counter(self.clusterNo)
			print('current class_count[-1]:',class_count[-1])
			print('class_count[-1]:',class_count[-1])
			
			
			
				
			if class_count[-1] < len(self.dataset)*0.2 or self.minpts==1:
				print('class_count[-1]:',class_count[-1])
				return 1
			self.minpts-=1
			return chunking()
		
		##collections.Counter():降順かつその値の数を返す {値:個数,値:個数,...} {key:value,...}
		class_count = collections.Counter(self.clusterNo)
		if 9999 in class_count:
			del class_count[9999]
		if -1 in class_count:
			del class_count[-1]
		##class_countにはクラスタされているデータインデックスが入っている
		print(class_count)
		n=no_of_cluster
		class_target=[]
		##lambda:無名関数 要素xを受け取り、-x[1]を返す
		##sorted(〇,key=lambda x = -x[1]):value降順ソート
		for i,(k, v) in enumerate(sorted(class_count.items(), key=lambda x: -x[1])):
			if n > i and k != 9999 and k != -1:
				class_target.append(k)
		print('class_target:',class_target)
		#ノイズ等は-1に返す
		self.clusterNo= [i if i in class_target else -1 for i in self.clusterNo]		
		chunking()
		#datasetを読んで該当クラスの場合のみ、expandclusterを実行。ただし、minptsを下げる（epsはそのまま）
		
		class_count = collections.Counter(self.clusterNo)
		print('final class_count[-1]:',class_count[-1])				
		return self.clusterNo,self.dataset
				
	def squeezing2(self,no_of_cluster,new_minpts,new_eps):
		self.eps = new_eps
		self.minpts=new_minpts
		def regionQuery(index):
			
			#距離計算する際，自分自身とも比較する
			#データ自体でなくて，datasetのindexを取得する
			neighbors = [ i for i,neighbor in enumerate(self.dataset) if np.linalg.norm(self.dataset[index]-neighbor)<=self.eps]
			return neighbors	
		
		
		def expandCluster(neighborPts,C):
			prev_neighborPts = neighborPts
			ex_neighborPts=[]
			
			#neighborPtsのindexが入っている
			for np_index in neighborPts:
			
				#print(np_index)
				if self.clusterNo[np_index] <0:
				#if self.clusterNo[np_index] ==-1 or self.clusterNo[np_index] ==-2:
					print('prev class',self.clusterNo[np_index])
					self.clusterNo[np_index] = C
					print('update class',self.clusterNo[np_index])
					#eps半径内にある任意の点np_indexを中心として，eps半径を描き，円内の点を探す その点のindexが返ってくる この時は，visitした点も改めてカウントする
					#visitした点が別のクラスタに属することもある					
					expand_neighborPts = regionQuery(np_index)
					#上記の円は領域拡大を意味する．minpts以上の点があれば領域を拡大を認める
					if len(expand_neighborPts) >= self.minpts:
						
						#visitしたポイントを除外して配列を作成する
						expand_neighborPts = [n_point for n_point in expand_neighborPts if self.clusterNo[n_point] == -1]
						
						ex_neighborPts+=expand_neighborPts
						
			
			if len(ex_neighborPts) == 0:
				
				return 1
				#ex_neighborPtsの平均的な数を計算し、それを大きく下回った閾値になったら脱出
			self.expand_counter+=1	
			return expandCluster(ex_neighborPts,C)
		
		#要素数の大きいクラスを上位nまで特定する
		def chunking():
			
			print('minpts:',self.minpts)
			for index,p in enumerate(self.dataset):
				#print(index,self.eps,self.minpts)
				neighborPts=[]
				C=0
				if self.clusterNo[index]!=-1:
					neighborPts = regionQuery(index)
				if len(neighborPts) >= self.minpts:
					self.expand_counter=0
					C=self.clusterNo[index]
					expandCluster(neighborPts,C)
				threshold_expand=5
				if self.expand_counter < threshold_expand:
					print(self.expand_counter)
					self.clusterNo= [-1 if i ==C else i for i in self.clusterNo]
					
			class_count = collections.Counter(self.clusterNo)
			print('current class_count[-1]:',class_count[-1])
			print('class_count[-1]:',class_count[-1])
			if class_count[-1] < len(self.dataset)*0.2 or self.minpts==1:
				print('class_count[-1]:',class_count[-1])
				return 1
			self.minpts-=1
			return chunking()
		
		
		
		self.clusterNo= [-1 if i ==9999 else i for i in self.clusterNo]		
		chunking()
		#datasetを読んで該当クラスの場合のみ、expandclusterを実行。ただし、minptsを下げる（epsはそのまま）
		
		class_count = collections.Counter(self.clusterNo)
		print('final class_count[-1]:',class_count[-1])				
		return self.clusterNo,self.dataset
				


