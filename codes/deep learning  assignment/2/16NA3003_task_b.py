
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


# In[2]:



d=DataLoader()
images_train,labels_train=d.load_data()
images_test,labels_test=d.load_data('test')
X_train, X_val, y_train, y_val = train_test_split(images_train, labels_train, test_size=0.30, random_state=42)


# In[3]:


from __future__ import print_function  
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn


# In[139]:





# In[189]:


net = nn.HybridSequential(prefix='MLP_')
with net.name_scope():
    net.add(
        nn.Flatten(),
        nn.Dense(1024, activation='relu'),
        #nn.Dropout(.6),
        nn.BatchNorm(),
        nn.Dense(512, activation='relu'),
        #nn.Dropout(.6),
        nn.BatchNorm(),
        nn.Dense(256, activation=None)

    )
    
ctx =   mx.gpu()#mx.gpu(0) if mx.test_utils.list_gpus() else
net.initialize(mx.init.Uniform(.1), ctx=ctx)


# In[190]:


trainer = gluon.Trainer(
    params=net.collect_params(),
    optimizer='Adam',
    optimizer_params={'learning_rate': 0.001},
)

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()


# In[191]:


batch_size=1024

X_test=mx.nd.array(images_test)
y_test=mx.nd.array(labels_test)

dataset=mx.gluon.data.dataset.ArrayDataset(X_train, y_train)
Val_set=mx.gluon.data.dataset.ArrayDataset(X_val, y_val)
test_set=mx.gluon.data.dataset.ArrayDataset(X_test, y_test)

train_loader=mx.gluon.data.DataLoader(dataset, shuffle='True', batch_size=batch_size)

valid_loader=mx.gluon.data.DataLoader(Val_set, shuffle='False', batch_size=batch_size)
Test_loader=mx.gluon.data.DataLoader(test_set, shuffle='False', batch_size=batch_size)


# In[192]:


def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


# In[193]:


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for data, label in data_iterator:
        #print(label)
        cumulative_loss = 0
        data, label = transform(data,label)
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        #print(output)
        loss = loss_function(output, label)
        cumulative_loss += nd.sum(loss).asscalar()
        predictions = nd.argmax(output, axis=1)
        #print (predictions)
        acc.update(preds=predictions, labels=label)
    return acc.get(), cumulative_loss


# In[194]:


def evaluate_accuracy_train(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data, label = transform(data,label)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# In[195]:


num_epochs = 50
number_ex=60000
Train_loss1=[]
Val_loss=[]
for epoch in range(num_epochs):
    sum_loss=0
    for inputs, labels in train_loader:
        #print(labels)
        inputs,labels = transform(inputs,labels)
        inputs = inputs.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        with autograd.record():
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
        loss.backward()
        metric.update(labels, outputs)
        sum_loss+=nd.sum(loss).asscalar()
        trainer.step(batch_size=inputs.shape[0])
   

    Train_loss1.append(sum_loss/number_ex)
    val_acc,val_loss=evaluate_accuracy(valid_loader,net)
    Val_loss.append(val_loss)

    name, acc = metric.get()
    print('After epoch {}: Training {} ={} Validation accuracy = {}'.format(epoch + 1, name, acc,val_acc))
    metric.reset()


# In[99]:


plt.figure("Image")
plt.title("Network 2 Loss vs Epoch")
Train_loss1=[]
for j in range(len(Train_loss)):
    Train_loss1.append(Train_loss[j]/np.sum(Train_loss))
Valai_loss1=[]
for i in range(len(Val_loss)):
    Val_loss1.append(Val_loss[i]/np.sum(Val_loss))

plt.plot(Train_loss1,c="red", label="Training Loss")
plt.plot(Val_loss1,c="green", label="Validation Loss")
plt.legend()
plt.figure("Image")





# In[198]:





# In[17]:





# In[199]:





# In[32]:




