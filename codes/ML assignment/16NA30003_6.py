#Roll-16NA30003
#NAME -Anmesh Choudhury	
#Assigment No-6
import numpy as np

def sigmoid(X):
	return 1./(1+np.exp(-X))

	
def sigmoid_p(X):
	return sigmoid(X) * (1-sigmoid(X))

def forwardprop(X,W):
	y=np.multiply(W,X)
	z=sigmoid(np.sum(y))
	return z
def backprop(Z,Y,X,W,learning_rate):
	cost = np.square(Z-Y)/2.
	dcost_prediction=(Z-Y)
    	dprediction_dz =sigmoid_p(Z)
    	W=W-learning_rate*np.multiply(np.multiply(dprediction_dz,dcost_prediction),X)
    	return W,cost

def Predict(X,W):
	a=[]
	for i in X:
		i=np.append(i,1)
		if(sigmoid(np.sum(np.multiply(i,W)))>=.5):
			a.append(1)
		else:
			a.append(0)
	return a
def train(X,Y,N):
	n=len(X[0])
	W=np.random.rand(n+1)
	x=[[]*9]*20
	for z in range(len(X)):
		 x[z]=np.insert(X[z],8,1)
	for j in range(10):
		for i in range(len(x)):
			Z=forwardprop(i,W)
			W,cost=backprop(Z,Y[i],x[i],W,N)#y has to index
			

	return W
def main():
 	data_set= np.genfromtxt('data6.csv', delimiter=",")
	data_set=np.array(data_set)
	X_train=data_set[:,:8]
	Y_train=data_set[:,8]
 	W=train(X_train,Y_train,.1)
 	X_test=np.genfromtxt('test6.csv', delimiter=",")
 	a=Predict(X_test,W)
 	f= open("16NA30003_6.out","w+")
 	for i in range(len(a)):
		f.write(str(a[i])+" ")
	f.close() 

if __name__ == '__main__':
	main()


 			



 		
