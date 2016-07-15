from net import SimpleCNN
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from gensim import corpora, matutils
from gensim.models import word2vec 

# Prepare dataset
X_train =
Y_train = 

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.15)
N_test = y_test.size         # test data size
N = len(x_train)             # train data size
in_units = x_train.shape[1]  #

# (nsample, channel, height, width)
input_channel = 1
x_train = x_train.reshape(len(x_train), input_channel, height, width) 
x_test  = x_test.reshape(len(x_test), input_channel, height, width)

n_units = args.nunits
n_label = 2
filter_height = 3
output_channel = 50

#モデルの定義
model = L.Classifier( SimpleCNN(input_channel, output_channel, filter_height, width, 950, n_units, n_label))

#GPUを使うかどうか
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
    perm = np.random.permutation(N) #ランダムな整数列リストを取得
    sum_train_loss     = 0.0
    sum_train_accuracy = 0.0
    for i in six.moves.range(0, N, batchsize):

        #perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]])) #source
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]])) #target

        optimizer.update(model, x, t)

        sum_train_loss      += float(model.loss.data) * len(t.data)   # 平均誤差計算用
        sum_train_accuracy  += float(model.accuracy.data ) * len(t.data)   # 平均正解率計算用

    print('train mean loss={}, accuracy={}'.format(sum_train_loss / N, sum_train_accuracy / N)) #平均誤差

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
        sum_test_loss / N_test, sum_test_accuracy / N_test)) #平均誤差

    if epoch > 10:
        optimizer.lr *= 0.97
        print 'learning rate: ', optimizer.lr

    sys.stdout.flush()
