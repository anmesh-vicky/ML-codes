
# coding: utf-8

# In[54]:
# In[56]:


from __future__ import print_function  
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn


# In[57]:
import sys
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


# In[61]:



d=DataLoader()
images_train,labels_train=d.load_data()
images_test,labels_test=d.load_data('test')
X_train, X_val, y_train, y_val = train_test_split(images_train, labels_train, test_size=0.30, random_state=42)


batch_size=1024



class Model(gluon.Block):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(512)
            self.dense1 = gluon.nn.Dense(128)
            self.dense2 = gluon.nn.Dense(64)
            self.dense3 = gluon.nn.Dense(32)
            self.dense4 = gluon.nn.Dense(32)
    def forward(self, x):
        x = mx.nd.relu(self.dense0(x))
        x = mx.nd.relu(self.dense1(x))
        x = mx.nd.relu(self.dense2(x))
        x = mx.nd.relu(self.dense3(x))
        
        return mx.nd.relu(self.dense4(x))


# In[58]:




X_test=mx.nd.array(images_test)
y_test=mx.nd.array(labels_test)

dataset=mx.gluon.data.dataset.ArrayDataset(X_train, y_train)
Val_set=mx.gluon.data.dataset.ArrayDataset(X_val, y_val)
test_set=mx.gluon.data.dataset.ArrayDataset(X_test, y_test)

train_loader=mx.gluon.data.DataLoader(dataset, shuffle='True', batch_size=batch_size)

valid_loader=mx.gluon.data.DataLoader(Val_set, shuffle='False', batch_size=batch_size)
Test_loader=mx.gluon.data.DataLoader(test_set, shuffle='False', batch_size=batch_size)


# In[63]:


def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


# In[64]:


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


# In[65]:


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


ctx =   mx.gpu(0) if mx.test_utils.list_gpus() else mx.cpu(0)
if(sys.argv[1]=='--train'):
    net=Model()
    
    net.initialize(mx.init.Uniform(0.1), ctx=ctx)
    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer='Adam',
        optimizer_params={'learning_rate': 0.001},
    )
    metric = mx.metric.Accuracy()
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    
    num_epochs = 50
    number_ex=60000
    Train_loss=[]
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
       

        Train_loss.append(sum_loss/number_ex)
        val_acc,val_loss=evaluate_accuracy(valid_loader,net)
        Val_loss.append(val_loss)
        name, acc = metric.get()
        print('After epoch {}: Training {} ={} Validation accuracy = {}'.format(epoch + 1, name, acc,val_acc))
        metric.reset()



    plt.figure("Image")
    plt.title("Network 2 Loss vs Epoch")
    Train_loss1=[]
    for j in range(len(Train_loss)):
        Train_loss1.append(Train_loss[j]/np.sum(Train_loss))
    Val_loss1=[]
    for i in range(len(Val_loss)):
        Val_loss1.append(Val_loss[i]/np.sum(Val_loss))

    plt.plot(Train_loss1,c="red", label="Training Loss")
    plt.plot(Val_loss1,c="green", label="Validation Loss")
    plt.legend()
    file_name = "net1.params"
    net.save_parameters(file_name)



elif(sys.argv[1]=='--test'):
    net = Model()
    net.load_parameters("net1.params")
    X=net.collect_params()
    cnt = 0
    accuracy = 0
    for data, label in Test_loader:
        data , label = transform(data,label)
        data = data.as_in_context(mx.cpu()).reshape((-1, 784))
        label = label.as_in_context(mx.cpu())
        with autograd.record():
            output = net(data)

            acc = mx.metric.Accuracy()
            acc.update(preds=nd.argmax(output,axis=1),labels=label)
            #print("Test Accuracy : %f"%acc.get()[1])
            accuracy = accuracy + acc.get()[1]
            cnt = cnt + 1
    print("Total Accuracy: ", float(accuracy/cnt))

