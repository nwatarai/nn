import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers
from zChainer import NNAutoEncoder, utility
from scipy import io
import sys

data = np.load(sys.argv[1])
k = int(sys.argv[2])

encoder = ChainList(
    L.Linear(4**k, 200),
    L.Linear(200, 100))

decoder =ChainList(
    L.Linear(200, 4**k),
    L.Linear(100, 200))

ae = NNAutoEncoder(encoder, decoder, optimizers.Adam(), epoch=100, batch_size=100,
    log_path=sys.argv[1]+".csv",
    export_path=sys.argv[1]+".model")

ae.fit(data)