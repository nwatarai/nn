import sys
import numpy as np

data = np.load(sys.argv[1])

for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		if data[i][j] == "0000":
			print "i=" + str(i) + ", j=" + str(j)