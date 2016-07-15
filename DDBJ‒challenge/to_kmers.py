import sys
import numpy as np

data = np.load(sys.argv[1])
k = int(sys.argv[2])

bases = ["1000","0100","0010","0001"]

def to_kmer_distribution(seq):
	distribution = np.zeros([seq.shape[0],4**k])
	for i in range(seq.shape[0]):
		for j in range(seq.shape[1] - k):
			try:
				kmer = seq[i][j:j+k]
				index = 0
				for b in range(k):
					index += (4**b) * bases.index(kmer[b])
				distribution[i][index] += 1
			except:
				continue
	return distribution

distribution = to_kmer_distribution(data)
np.save(sys.argv[1]+"."+sys.argv[2]+"mer",distribution)
