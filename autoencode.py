from sklearn.datasets import fetch_mldata
from PIL import Image
import numpy as np
import pandas as pd
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers
from zChainer import NNAutoEncoder, utility
from MNIST import imshow, imshow_all, sampling

X, t = sampling(50000)

# 784-200-100 nodes network
encoder = ChainList(
    L.Linear(784, 200),
    L.Linear(200, 100))
decoder =ChainList(
    L.Linear(200, 784),
    L.Linear(100, 200))

'''
encoder = ChainList(
    L.Linear(784, 300))
decoder =ChainList(
    L.Linear(300, 784))
'''
ae = NNAutoEncoder(encoder, decoder, optimizers.Adam(), epoch=100, batch_size=100,log_path="./ae_"+utility.now()+"_log.csv", export_path="./ae_"+utility.now()+".model")
#ae = NNAutoEncoder(encoder, decoder, optimizers.Adam(), epoch=10, batch_size=100)

X = X.astype(np.float32)
ae.fit(X)
'''
print dir(ae.model[0])
print ae.model[0].W.data.shape
print ae.model[1].W.data.shape
print len(ae.model)
'''