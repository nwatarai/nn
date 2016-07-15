import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers, Variable
from chainer.functions.activation.softmax import softmax
from zChainer import NNManager, utility
import pickle
import sys
from scipy import io

data = io.loadmat(sys.argv[1])
kmer_tr = np.load(sys.argv[2])
#kmer_te = np.load(sys.argv[3])
f = open(sys.argv[4])
model = pickle.load(f)
f.close()

X_train = kmer_tr.astype(np.float32)
y_train = data['out_tr'].astype(np.int32)
X_test = kmer_te.astype(np.float32)
model.add_link(L.Linear(100,8))

def forward(self, x):
    h = x
    for layer in xrange(len(model)):
        h = F.dropout(F.relu(self.model[layer](h)),ratio=0.6,train=True)
    return h

def output(self, y):
    y_trimed = y.data.argmax(axis=1)
    return np.array(y_trimed, dtype=np.int32)

NNManager.forward = forward
NNManager.output = output
nn = NNManager(model, optimizers.Adam(), F.softmax_cross_entropy,
    epoch=100, batch_size=100,
    log_path=sys.argv[2]+".csv",
    export_path=sys.argv[2]+".model")

nn.fit(X_train, y_train, is_classification=True)

def _forward(x):
    h = Variable(x)
    for layer in xrange(len(model)-1):
        h = F.relu(model[layer](h))
    return softmax(model[-1](h)).data

ans = _forward(X_test)
np.savetxt(sys.argv[2]+".tsv",ans,delimiter="\t", fmt="%.6f")