from sklearn.datasets import fetch_mldata
from PIL import Image
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers, ChainList
import chainer.functions as F
from chainer.functions.activation.softmax import softmax
import chainer.links as L
from zChainer import NNManager, utility
import pickle
from MNIST import imshow, imshow_all, sampling
import sys

#model = ChainList(L.Linear(784, 200), L.Linear(200, 100), L.Linear(100, 10))
# or load a serialized model

f = open(sys.argv[1])
model = pickle.load(f)
f.close()
model.add_link(L.Linear(100,10))

batchsize = 100
n_epoch   = 30
N = 10000
x, t = sampling(N)
x_train = x.astype(np.float32)
y_train = t.astype(np.int32)
N_test = 100
x, t = sampling(N_test)
x_test = x.astype(np.float32)
y_test = t.astype(np.int32)

quit()
#------------------------------
"""

def forward(x_data,t_data,train=False):
    h = Variable(x_data)
    t = Variable(t_data)
    for layer in range(len(model)-1):
        h = F.dropout(F.relu(model[layer](h)), ratio=0.6, train=train)
    y = model[-1](h)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def output(y):
    y_trimed = y.data.argmax(axis=1)
    return np.array(y_trimed, dtype=np.int32)

def _output(y):
    y_trimed = y.data.argmax(axis=1)
    out = np.zeros([y.data.shape[0], y.data.shape[1]])
    for i in xrange(out.shape[0]):
        out[i][y_trimed[i]] = 1
    return np.array(out, dtype=np.int32)
#set up
optimizer = optimizers.Adam()
optimizer.setup(model)

train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

lW = [[] for i in range(len(model))]

# Learning loop
for epoch in xrange(n_epoch):
    print 'epoch', epoch

    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

    optimizer.zero_grads()
    loss, acc = forward(x_batch, y_batch, train=True)
    loss.backward()
    optimizer.update()

    train_loss.append(loss.data)
    train_acc.append(acc.data)
    sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
    sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)

    sum_accuracy = 0
    sum_loss     = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]

    loss, acc = forward(x_batch, y_batch, train=False)

    test_loss.append(loss.data)
    test_acc.append(acc.data)
    sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
    sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)

    for i in xrange(len(model)):
        lW[i].append(model[i].W)

quit()
"""
#-----------------------------------

def forward(self, x):
    h = x
    for layer in xrange(len(self.model)):
        h = F.relu(self.model[layer](h))
    return h

def output(self, y):
    y_trimed = y.data.argmax(axis=1)
    return np.array(y_trimed, dtype=np.int32)

NNManager.forward = forward
nn = NNManager(model, optimizers.Adam(), F.softmax_cross_entropy,
    epoch=n_epoch, batch_size=batchsize,
    log_path="model_trained.csv",
    export_path="model_trained.model")

nn.fit(x_train, y_train, x_test, y_test, is_classification=True)

"""

def _forward(x_data):
    h = Variable(x_data)
    for layer in range(len(model)-1):
        h = F.relu(model[layer](h))
    return softmax(model[-1](h)).data

np.savetxt("test.tsv", _forward(x_test), delimiter="\t", fmt='%.6f')
"""