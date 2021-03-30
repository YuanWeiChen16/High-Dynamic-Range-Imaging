import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

x_max = 2112
y_max = 2816
imageNum = 4
sampleNum = 50


img = np.ndarray(shape = (imageNum, x_max, y_max, 3), dtype = int)
img[0] = mpimg.imread('image+4.jpg')
img[1] = mpimg.imread('image+2.jpg')
img[2] = mpimg.imread('image-2.jpg')
img[3] = mpimg.imread('image-4.jpg')

random.seed(a = None, version = 2)
idx = np.random.randint(x_max, size = sampleNum)
idy = np.random.randint(y_max, size = sampleNum)
index = np.random.uniform(low = [0, 0], high = [x_max, y_max], size = (sampleNum, 2))

Z = np.ndarray(shape = (sampleNum, imageNum, 3))

for i in range(0, sampleNum):
    for j in range(0, imageNum):
        Z[i][j] = img[j][idx[i]][idy[i]]
