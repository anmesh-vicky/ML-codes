
# coding: utf-8

# In[1]:


from zipfile import ZipFile
import numpy as np
from matplotlib import pyplot as plt
'''load your data here'''
from sklearn.model_selection import train_test_split

class DataLoader(object):
    def __init__(self):
        DIR = "../data/"
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

    def create_batches(self):
        pass


# In[3]:



d=DataLoader()
images_train,labels_train=d.load_data()
images_test,labels_test=d.load_data('test')
X_train, X_val, y_train, y_val = train_test_split(images_train, labels_train, test_size=0.30, random_state=42)


# In[ ]:


batch_size=1024

X_test=mx.nd.array(images_test)
y_test=mx.nd.array(labels_test)

dataset=mx.gluon.data.dataset.ArrayDataset(X_train, y_train)
Val_set=mx.gluon.data.dataset.ArrayDataset(X_val, y_val)
test_set=mx.gluon.data.dataset.ArrayDataset(X_test, y_test)

train_loader=mx.gluon.data.DataLoader(dataset, shuffle='True', batch_size=batch_size)

valid_loader=mx.gluon.data.DataLoader(Val_set, shuffle='False', batch_size=batch_size)
Test_loader=mx.gluon.data.DataLoader(test_set, shuffle='False', batch_size=batch_size)


# In[4]:


from __future__ import print_function  
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn


# In[6]:



class Model(gluon.Block):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(1024)
            self.dense1 = gluon.nn.Dense(512)
            self.mydense = gluon.nn.Dense(256, prefix='mydense_')

    def forward(self, x):
        x = mx.nd.relu(self.dense0(x))
        x = mx.nd.relu(self.dense1(x))
        return mx.nd.relu(self.mydense(x))


# In[7]:


net=Model()
ctx =mx.gpu(0) if mx.test_utils.list_gpus() else mx.cpu(0)


# In[8]:


net.load_parameters("net.params")
X=net.collect_params()


# In[13]:


W0=X['model0_dense0_weight'].data().as_in_context(ctx)
b0=X['model0_dense0_bias'].data().as_in_context(ctx)


# In[ ]:


Z=nd.dot(mx.nd.array(X_train).as_in_context(ctx),W0)+b0


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lm = LogisticRegression(C= 0.01, solver: 'lbfgs')
lm.fit(Z,y_train)
#predicting on test set
y_pred = lm.predict(X_test)
accuracy_score(y_true, y_pred)

