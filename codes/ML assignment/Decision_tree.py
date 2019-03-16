#
import numpy as np 
def entropy_feature(data,n):
	sum1=0
	sum2=0
	len1=0
	len2=0
	ratio1=0
	ratio2=0
	if data.ndim==1:
		return 0;

	for i in range(len(data)):
		if data[i][n]==1:
			sum1=sum1+data[i][len(data[0])-1]
			len1=len1+1
		else :
			sum2=sum2+data[i][len(data[0])-1]
	
	if(len1!=0):
		ratio1=sum1/len1
	if (len2!=0):
		ratio2=sum2/len2
	elif (ratio1==0 or ratio2==1) and (ratio2!=0 and ratio2!=1):
		entropy=(-1*ratio2*np.log2(ratio2)-1*(1-ratio2)*np.log2(1-ratio2))*len2/len(data)
	elif (ratio2==0 or ratio2==1) and (ratio1!=0 and  ratio1!=1):
		entropy=(-1*ratio1*np.log2(ratio1)-1*(1-ratio1)*np.log2(1-ratio1))*len1/len(data)
	elif (ratio2==0 or ratio2==1) and (ratio1==0 or ratio1==1):
		return 0
	else:
		entropy=len1/len(data)*(-1*ratio1*np.log2(ratio1)-1*(1-ratio1)*np.log2(1-ratio1))+(-1*ratio2*np.log2(ratio2)-1*(1-ratio2)*np.log2(1-ratio2))*len2/len(data)
	
	return entropy

def entropy(data ,n):
	if data.ndim==1:
		postive=data[n]
	else:
		postive=(np.sum(data,axis=0))[n]
	total=len(data)
	ratio =postive/total
	if ratio==0:
		return 0
	elif ratio==1:
		return 0
	entropy=-1*ratio*np.log2(ratio)-1*(1-ratio)*np.log2(1-ratio)
	
	return entropy


def gain(data,j):
	if data.ndim==1:
		n=len(data)-1
	else:
		n=len(data[0])-1

	total_entropy=entropy(data ,n)

	

	gain=total_entropy-entropy_feature(data,j)


	return gain












class tree:## binary treeeeeee left =0 right =1
	def __init__(self):
		self.left=None
		self.right=None
		self.data=None



def Decision_tree(csv,Tree):
		if csv.ndim==1:
			n=len(csv)-1
		else:
			n=len(csv[0])-1
		Gain=[]
		
		for j in range(n):
			
			Gain.append(gain(csv,j))

		
		Max=Gain.index(max(Gain))
		Min=Gain.index(min(Gain))
		
		right=np.array([])
		left=np.array([])
		Tree.data=Max
	
		Tree.right=tree()
		Tree.left=tree()
		if csv.ndim==2:
			for i in range(len(csv)):
		 		if len(right)==0 :
		 			if (csv[i][Max]==1):
						right=np.hstack((right,csv[i]))

				elif len(left)==0:
					if (csv[i][Max]==0):
						left=np.hstack((left,csv[i]))

				else:
					if (csv[i][Max]==1):
							right=np.vstack((right,csv[i]))
					else:
							left=np.vstack((left,csv[i]))

		if len(left)!=0:
			if entropy(left,n)==0:

				if left[0][n]==1:
					Tree.left=1
					
					
				else:
					Tree.left=0
					
			else:

				try:
				 	left=np.delete(left,Max,1)
				 	Decision_tree(left,Tree.left)
				except:
					left=np.delete(left,Max)
					Decision_tree(left,Tree.left)
		else:
			if right[0][n]==1:
				Tree.left=0
			else:
				Tree.left=1

		if len(right)!=0:			
			if entropy(right,n)==0:
				
				if right[0][n]==1:
					Tree.right=1
				
					
				else:
					Tree.right=0
					
			else:
				try:
					right=np.delete(right,Max,1)
					Decision_tree(right,Tree.right)
			        except:
					right=np.delete(right,Max)
		 			Decision_tree(right,Tree.right)
		else:
			if left[0][n]==1:
				Tree.right=0
			else:
				Tree.right=1
		
def test(Tree,Data):
	while(True):
		if Data[Tree.data]==1:
			if Tree.right==1:
				return 1
			elif Tree.right==0:
				return 0
			else:
				
				Data=np.delete(Data,Tree.data)
				Tree=Tree.right
				
		else:
			if Tree.left==1:
				return 1
			elif Tree.left==0:
				return 0
			else:
				
				Data=np.delete(Data,Tree.data)
				Tree=Tree.left




Root=tree()
def main():
	csv = np.genfromtxt('data2.csv', delimiter=",")
	csv=np.array(csv)
	
	Decision_tree(csv,Root)
	csv1 = np.genfromtxt('test2.csv', delimiter=",")
	
	
	for i in range(len(csv1)):
	 	print test(Root,csv1[i]),
	
	
	


if __name__ == '__main__':
	main()

