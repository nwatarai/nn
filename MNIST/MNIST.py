from sklearn.datasets import fetch_mldata
from PIL import Image
import numpy as np
import pandas as pd

def imshow(vector,x=28,y=28,revise=True):
	if revise:
		vector = np.array(vector)
		vmax = vector.max()
		vmin = vector.min()
		vector = (vector - vmin) / vmax * 255
	pilimg = Image.fromarray(vector.reshape(x,y))
	pilimg.show()

def imshow_all(vectors,x=28,y=28,column=20,revise=True,save=False):
	length = vectors.shape[0]
	_x = x + int(x/10)
	_y = y + int(y/10)
	X = column * _x
	Y = length / column * _y
	canvas = Image.new('RGB', (X, Y), (255, 255, 255))
	for i, vector in enumerate(vectors):
		if revise:
			vector = np.array(vector)
			vmax = vector.max()
			vmin = vector.min()
			vector = (vector - vmin) / vmax * 255
		pilimg = Image.fromarray(vector.reshape(x,y))
		canvas.paste(pilimg, ( (i%column)*_x , int(i/column)*_y ))
	canvas.show()
	if save:
		canvas.save('canvas.png')

def sampling(data_size):
	mnist = fetch_mldata('MNIST original', data_home='.')
	# X=matrix, t=teacher
	X, _, _, t = mnist.values()
	index = np.arange(70000)
	np.random.shuffle(index)
	_X = X[index[:data_size]]
	_t = t[index[:data_size]]
	return _X, _t

'''
#correlation between variance and bias
V = np.var(model[0].W.data,axis=1)
#index = range(200)
#index.sort(key=lambda x: V[x])
#print index

B = np.absolute(model[0].b.data)
#b_index = range(200)
#b_index.sort(key=lambda x: B[x])
#print b_index

from scipy.stats import spearmanr
print spearmanr(V,B)
'''