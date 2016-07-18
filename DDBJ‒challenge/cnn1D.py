from net import SimpleCNN
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from scipy import io
import sys
import pickle
data = io.loadmat(sys.argv[1])
#print data['seq_tr'].shape
#print data['seq_te'].shape
#print data['out_tr'].shape

# args
parser = argparse.ArgumentParser()
parser.add_argument('--input  '    , dest='input'        , type=str, default=None,            help='input file')
parser.add_argument('--gpu  '    , dest='gpu'        , type=int, default=0,            help='1: use gpu, 0: use cpu')
parser.add_argument('--data '    , dest='data'       , type=str, default='input.dat',  help='an input data file')
parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=100,          help='number of epochs to learn')
parser.add_argument('--batchsize', dest='batchsize'  , type=int, default=100,           help='learning minibatch size')
parser.add_argument('--nunits'   , dest='nunits'     , type=int, default=400,          help='number of units')
parser.add_argument('--k'   , dest='k'     , type=int, default=4,          help='frame size(k of k_mer)')

args = parser.parse_args()
batchsize   = args.batchsize  
n_epoch     = args.epoch       

# Prepare dataset
X_train = data['seq_tr'].astype(np.float32)
Y_train = data['out_tr'].argmax(axis=1).astype(np.int32)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.15)
N_test = y_test.size         # test data size
N = len(x_train)             # train data size
in_units = x_train.shape[1]  # number of samples

# (nsample, channel, height, width)
input_channel = 1
height = 200 # base pairs
width = 4 # sort of bases (ATGC)
x_train = x_train.reshape(len(x_train), input_channel, height, width) 
x_test  = x_test.reshape(len(x_test), input_channel, height, width)

n_units = args.nunits
n_label = np.amax(Y_train) + 1 # number of recognition target
filter_height = args.k # k of kmer
output_channel = 50
n_kernel = 1 # skipping frame size for max pooling
mid_units = int((height / n_kernel) - max(0, int((filter_height - n_kernel)/n_kernel))) * output_channel

model = L.Classifier( SimpleCNN(input_channel, output_channel, filter_height, width, mid_units, n_units, n_label, n_kernel))

if args.gpu > 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu <= 0 else cuda.cupy #args.gpu <= 0: use cpu, otherwise: use gpu

batchsize = args.batchsize
n_epoch = args.epoch

# Setup optimizer
optimizer = optimizers.AdaGrad()
optimizer.setup(model)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):

    print 'epoch', epoch, '/', n_epoch

    # training)
    perm = np.random.permutation(N) 
    sum_train_loss     = 0.0
    sum_train_accuracy = 0.0
    for i in six.moves.range(0, N, batchsize):

        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]])) #source
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]])) #target

        optimizer.update(model, x, t)

        sum_train_loss      += float(model.loss.data) * len(t.data)   
        sum_train_accuracy  += float(model.accuracy.data ) * len(t.data)  

    print('train mean loss={}, accuracy={}'.format(sum_train_loss / N, sum_train_accuracy / N)) 

    # evaluation
    sum_test_loss     = 0.0
    sum_test_accuracy = 0.0
    for i in six.moves.range(0, N_test, batchsize):

        # all test data
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))

        loss = model(x, t)

        sum_test_loss     += float(loss.data) * len(t.data)
        sum_test_accuracy += float(model.accuracy.data)  * len(t.data)

    print(' test mean loss={}, accuracy={}'.format(
        sum_test_loss / N_test, sum_test_accuracy / N_test)) 

    if epoch > 10:
        optimizer.lr *= 0.97
        print 'learning rate: ', optimizer.lr

model.to_cpu()
pickle.dump(model, open(args.input+"cnn.model", 'wb'), -1)

