import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import math
import cv2
x_max = 480
y_max = 640
imageNum = 4
sampleNum = 50

def weight(z):
    z_max, z_min = 255., 0. 
    if z <= ((z_max + z_min) / 2):
        return z - z_min
    else:
        return z_max - z

def estimate_curve(Z, exp, lamdba):
    z_max, z_min = 255., 0. 
    n = 255
    A = np.zeros(shape = ((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0])), dtype=float)
    b = np.zeros(shape = (A.shape[0], 1), dtype = float)

    k = 0
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            wij = weight(Z[i][j])
            A[k][Z[i][j]] = wij
            A[k][n + i] = -wij
            b[k][0] = wij * exps[j]
            k = k + 1

    A[k][int((z_max - z_min) * 0.5)] = 1
    k = k + 1

    for i in range(0, n - 2):
        A[k][i] = lamdba * weight(i + 1)
        A[k][i + 1] = -2 * lamdba * weight(i + 1)
        A[k][i + 2] = lamdba * weight(i + 1)
        k = k + 1
    
    x, _, _, _ = np.linalg.lstsq(A, b)
    g = x[0: n + 1]
    lE = x[n + 1: x.shape[0]]
    #print(g)
    return g[:,0], lE

def estimate_radiance(imgs, exps, curve):
    img_shape = imgs.shape
    rad = np.zeros(shape=img_shape[1:], dtype=float)

    imgNums = imgs.shape[0]

    for i in range(0, img_shape[1]):
        for j in range(0, img_shape[2]):
            g = np.ndarray(shape = imgNums, dtype=float)
            w = np.ndarray(shape = imgNums, dtype=float)
            for k in range(0, imgNums):
                g[k] = curve[int(imgs[k][i][j])]
                w = weight(imgs[k][i][j])

            sumOfW = np.sum(w)
            if sumOfW > 0:
                rad[i][j] = np.sum(w * (g - exps) / sumOfW)
            else:
                rad[i][j] = g[imgNums // 2] - exps[imgNums //2]
    return rad

img = np.ndarray(shape = (imageNum, x_max, y_max, 3), dtype = int)
exps = np.ndarray(shape = (imageNum), dtype = float)
img[0] = mpimg.imread('image+4m.jpg')
img[1] = mpimg.imread('image+2m.jpg')
img[2] = mpimg.imread('image-2m.jpg')
img[3] = mpimg.imread('image-4m.jpg')
exps[0] = (pow(2, 4))
exps[1] = (pow(2, 2))
exps[2] = (pow(2, -2))
exps[3] = (pow(2, -4))

random.seed(a = None, version = 2)
idx = np.random.randint(x_max, size = sampleNum)
idy = np.random.randint(y_max, size = sampleNum)
index = np.random.uniform(low = [0, 0], high = [x_max, y_max], size = (sampleNum, 2))

Z = np.ndarray(shape = (3, sampleNum, imageNum), dtype = int)

for i in range(0, sampleNum):
    for j in range(0, imageNum):
        Z[0][i][j] = img[j][idx[i]][idy[i]][0]
        Z[1][i][j] = img[j][idx[i]][idy[i]][1]
        Z[2][i][j] = img[j][idx[i]][idy[i]][2]
hdr_img = np.ndarray(shape = (img[0].shape), dtype = float)
for i in range(0, 3):
    g, lE = estimate_curve(Z[0], exps, 1)
    print(g.shape)
    rad = estimate_radiance(img[:,:,:,i], exps, g)
    print(rad)
    hdr_img[:,:,i] = cv2.normalize(rad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

from matplotlib.pylab import cm
colorize = cm.jet
cmap = np.float32(cv2.cvtColor(np.uint8(hdr_img), cv2.COLOR_BGR2GRAY)/255.)
cmap = colorize(cmap)
cv2.imwrite('cmap.jpg', np.uint8(cmap*255.))
