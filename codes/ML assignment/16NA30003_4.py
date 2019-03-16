#16NA30003
#Anmesh Choudhury
#assignment no-4

import math
import numpy as np
import operator


def distance(data,predict,k=5):
	distance=[]
	result=[]
	for i in data:
		dist=0
		for j in range(len(predict)):
			dist+=(i[j]-predict[j])**2
		dist=math.sqrt(dist)
		distance.append((dist,i[len(data[0])-1]))
	distance.sort(key=operator.itemgetter(0))#sorting accorting to distance
	for i in range(k):#taking Top a K neighbours
		result.append(distance[k][1])
	return result


def main():
	data=np.genfromtxt('data4.csv', delimiter=",")
	data_predict=np.genfromtxt('test4.csv', delimiter=",")
	f= open("16NA30003_4.out","w+")
	for i in data_predict:
		result=distance(data,i,5)
		if np.sum(result)>2:
			f.write(str(1)+" ")
		else :
			f.write(str(0)+" ")
if __name__ == '__main__':
	main()
