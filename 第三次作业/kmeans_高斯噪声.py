import random 
import pandas as pd
import numpy as np




def circle_o (df,cent1,cent2,cent3,list_sum,list1,list2,list3,result):
	for j in range(df.shape[0]):
		for i in range(df.shape[1]):
			a = np.abs(df[i][j]-cent1)
			b = np.abs(df[i][j]-cent2)
			c = np.abs(df[i][j]-cent3)

			if a<=b and a<=c:
				list1.append(df[i][j])
			if b<=a and b<=c:
				list2.append(df[i][j])
			if c<=a and c<=b:
				list3.append(df[i][j])

	nd1 = np.array(list1)
	nd2 = np.array(list2)
	nd3 = np.array(list3)

	n1 = len(list1)
	n2 = len(list2)
	n3 = len(list3)

	list_sum.clear()
	for num in list1:
		s1 = abs(num*n1 - nd1.sum())
		list_sum.append(s1)
	index_m = np.array(list_sum).argmin()
	cent_n1 = list1[index_m]

	list_sum.clear()
	for num in list2:
		s2 = abs(num*n2 - nd2.sum())
		list_sum.append(s2)
	index_m = np.array(list_sum).argmin()
	cent_n2 = list2[index_m]

	list_sum.clear()
	for num in list3:
		s3 = abs(num*n3 - nd3.sum())
		list_sum.append(s3)
	index_m = np.array(list_sum).argmin()
	cent_n3 = list3[index_m]

	if cent_n1 != cent1:
		cent1 = cent_n1
	if cent_n2 != cent2:
		cent2 = cent_n2
	if cent_n3 != cent3:
		cent3 = cent_n3

	if cent_n1 == cent1 and cent_n2 == cent2 and cent_n3 == cent3:
		result = 0
	else:
		result = 1

	return result

#设置精度小数点后两位
np.set_printoptions(precision=2)

#读取文件
df=pd.read_csv("waveform.data",header = None)

#增加20%的高斯噪声
for x in range(5000*22//5):
	i = random.randint(0,21)
	j = random.randint(0,4999)
	df[i][j] += random.gauss(0,0.5)  #均值维0，方差为0.5的高斯噪声


#随机选择三个中心点作为初始质心
i1 = random.randint(0,21)     #列标0-21
j1 = random.randint(0,4998)   #行标0-4999
cent1 = df[i1][j1]

i2 = random.randint(0,21)     
j2 = random.randint(0,4998)   
cent2 = df[i2][j2]

i3 = random.randint(0,21)     #列标0-21
j3 = random.randint(0,4998)   #行标0-4999
cent3 = df[i3][j3]

#聚类存储列表
list1 = []
list2 = []
list3 = []
list_sum = []

result = 1

while result:
	list1.clear()
	list2.clear()
	list3.clear()
	result = circle_o(df,cent1,cent2,cent3,list_sum,list1,list2,list3,result)
	

print("Cluster 1:",list1[:100],'\n')
print("Cluster 2:",list2[:100],'\n')
print("Cluster 3:",list3[:100],'\n')