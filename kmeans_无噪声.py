import random 
import pandas as pd
import numpy as np

'''
cluster函数用于判别各条数据所属类别，循环更新至聚类中心不变为止
'''
def cluster (df,c1,c2,c3,list1,list2,list3,result):
	for j in range(df.shape[0]):
		for i in range(df.shape[1]):
			a = np.abs(df[i][j]-c1)
			b = np.abs(df[i][j]-c2)
			c = np.abs(df[i][j]-c3)

			if a<b and a<c:
				list1.append(df[i][j])
			if b<a and b<c:
				list2.append(df[i][j])
			if c<a and c<b:
				list3.append(df[i][j])
			if a==b and a==c:
				list1.append(df[i][j])
                
	nd1 = np.array(list1)
	nd2 = np.array(list2)
	nd3 = np.array(list3)

	c_n1 = round(np.mean(nd1),2)
	c_n2 = round(np.mean(nd2),2)
	c_n3 = round(np.mean(nd3),2)

	if c_n1 != c1:
		c1 = c1
	if c_n2 != c2:
		c2 = c_n2
	if c_n3 != c3:
		c3 = c_n3

	if c_n1 == c1 and c_n2 == c2 and c_n3 == c3:
		result = 0
	else:
		result = 1
	return result


np.set_printoptions(precision=2)



#输入数据集开始处理
df=pd.read_csv("waveform.data",header = None)
i1 = random.randint(0,20)    
j1 = random.randint(0,4998)
c1 = df[i1][j1]
i2 = random.randint(0,20)     
j2 = random.randint(0,4998)   
c2 = df[i2][j2]
i3 = random.randint(0,20)
j3 = random.randint(0,4998)
c3 = df[i3][j3]
list1 = []
list2 = []
list3 = []
result = 1
while result:
	list1.clear
	list2.clear
	list3.clear
	result = cluster(df,c1,c2,c3,list1,list2,list3,result)

print("Cluster 1:",list1[:100],'\n')
print("Cluster 2:",list2[:100],'\n')
print("Cluster 3:",list3[:100],'\n')
