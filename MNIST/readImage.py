from PIL import Image
import sys
import numpy as np

img = Image.open(sys.argv[1])

nimg = np.array(img)

print(nimg.shape)