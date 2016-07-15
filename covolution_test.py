import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers, Variable
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split


def reshape_for_Convolution2D(x_data, IC, OC, kernel):
    B = x_data.shape[0]
    L = x_data.shape[1]
    s = Variable(x_data.astype(np.float32))
    l = chainer.links.Convolution2D(IC, OC, (kernel, 1))
    s = chainer.functions.reshape(s, (B, IC, -1, 1))
    x = l(s)
    x = chainer.functions.reshape(x, (B, OC, -1))
    return x

quit() 

mnist=fetch_mldata('MNIST original',data_home=".")
x=mnist.data
y=mnist.target
x=x.astype(np.float32)
y=y.astype(np.int32)
  
#x /= x.max()
  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
  
N=y_train.size   
N_test=y_test.size  
print("train data=%d" % N)
print("test data=%d" % N_test)
  
#[#image][#color][height][width]
x_train=x_train.reshape((len(x_train),1,28,28))
x_test=x_test.reshape((len(x_test),1,28,28))
  
batchsize=100 
n_epoch=20   
n_units=100     
  
model=chainer.FunctionSet( 
    conv1=F.Convolution2D(1,5,3,pad=0),#28x28x1 -> 26x26x5
    #Max Pooling: 26x26x5 -> 13x13x5
    l1=F.Linear(13*13*5,n_units), 
    l2=F.Linear(n_units,10))
  

def forward(x_data,y_data,train=True):
    x=chainer.Variable(x_data)
    t=chainer.Variable(y_data)
    
    h0=F.max_pooling_2d(F.relu(model.conv1(x)),2)
    h1=F.dropout(F.relu(model.l1(h0)),train=train)
    y=model.l2(h1)
    
    if train:
        loss=F.softmax_cross_entropy(y,t)
        return loss
    else:
        acc=F.accuracy(y,t)
        return acc
  
optimizer=optimizers.Adam()
optimizer.setup(model)
  
for epoch in range(1,n_epoch+1):
    print("epoch: %d" % epoch)
    perm=np.random.permutation(N)
    sum_loss=0
    for i in range(0,N,batchsize):
        x_batch=np.asarray(x_train[perm[i:i+batchsize]])
        y_batch=np.asarray(y_train[perm[i:i+batchsize]])
        
        optimizer.zero_grads()
        loss=forward(x_batch,y_batch)
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data)*len(y_batch)
    print("train mean loss: %f" % (sum_loss/N))
    
    sum_accuracy=0
    for i in range(0,N_test,batchsize):
        x_batch=np.asarray(x_test[i:i+batchsize])
        y_batch=np.asarray(y_test[i:i+batchsize])

        acc=forward(x_batch,y_batch,train=False)

        sum_accuracy += float(acc.data)*len(y_batch)
    print("test accuracy: %f" % (sum_accuracy/N_test))

import cPickle
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)