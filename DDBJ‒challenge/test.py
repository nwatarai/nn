from scipy import io
import sys
import numpy as np
data = io.loadmat(sys.argv[1])

#print data['seq_tr'].shape
#print data['seq_te'].shape
#print data['out_tr'].shape

def to_bases(key):
	d = data[key]
	out = [[] for i in range(d.shape[0])]
	for i in range(d.shape[1]/4):
		start = i*4 
		end = i*4 + 4
		ex = d[:,start:end]
		for j in range(d.shape[0]):
			out[j].append("".join([str(ex[j][i]) for i in range(4)]))
	np.save(sys.argv[1]+"."+key, np.array(out))

for i in ["seq_tr", "seq_te"]:
	to_bases(i)