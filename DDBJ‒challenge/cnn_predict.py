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
f = open(sys.argv[2])
model = pickle.load(f)
f.close()

input_channel = 1
height = 200 # base pairs
width = 4 # sort of bases (ATGC)
X_test = data['seq_te'].astype(np.float32)
X_test = X_test.reshape(len(X_test), input_channel, height, width)

x = chainer.Variable(X_test)
ans = model.predictor.predict(x)

out = ans.data
np.savetxt(sys.argv[2]+".tsv",out,delimiter="\t", fmt="%.6f")
